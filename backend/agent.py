"""
AskAra+ — LangGraph ReAct Agent (MCP-aware)
--------------------------------------------
Tools are called via the FastMCP server running at MCP_SERVER_URL
(default: http://localhost:8001/mcp).  If the MCP server is unreachable,
every tool call automatically falls back to direct Python imports — so
the agent works in both single-process and two-process deployments.

Process topology:
    [mcp_server.py :8001]  ←── MCP HTTP ───  [agent.py inside server.py :8000]
         FastMCP                                  LangGraph ReAct graph

Startup sequence (called by server.py lifespan):
    await init_mcp_tools()   → connects to MCP, loads tool objects once

Shutdown:
    await cleanup_agent()    → closes MCP connection

Architecture (LangGraph StateGraph):
    detect_context → react_agent → [tool_executor → react_agent]* → END

WebSocket event protocol:
    {"type": "reasoning",  "content": "<thought>"}      ← new
    {"type": "tool_start", "content": "<tool name>"}
    {"type": "tool_end",   "content": "<tool name>"}
    {"type": "structured", "content": {…}}
    {"type": "sources",    "content": […]}
    {"type": "token",      "content": "<token>"}
    {"type": "done",       "content": "<full response>"}
    {"type": "error",      "content": "<message>"}
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncGenerator, Optional, TypedDict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("askara.agent")

# ── Direct tool imports (fallback when MCP is unavailable) ───────────────────
from tools.search import search_documents as _search_documents
from tools.language import detect_language as _detect_language
from tools.simplify import simplify_text as _simplify_text
from tools.translate import translate_text as _translate_text
from tools.summarize import summarize_text as _summarize_text
from tools.dialect import dialect_adapt as _dialect_adapt
from tools.portal import fetch_gov_portal as _fetch_gov_portal
from tools.profiler import profile_match as _profile_match

from llm_client import call_llm, call_llm_streaming, call_llm_vision_streaming, LLMError

# ── LangGraph ─────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END

# ── System Prompt ─────────────────────────────────────────────────────────────
_PROMPT_CANDIDATES = [
    Path(__file__).parent / "system_prompt.txt",
    Path(__file__).parent / "prompts" / "system_prompt.txt",
]
SYSTEM_PROMPT = next(
    (p.read_text(encoding="utf-8") for p in _PROMPT_CANDIDATES if p.exists()),
    (
        "You are Ara, a warm multilingual assistant for AskAra+. "
        "Help ASEAN migrant workers and vulnerable populations access "
        "government programs and services in simple, clear language."
    ),
)

# ── MCP Configuration ─────────────────────────────────────────────────────────
MCP_SERVER_URL = "http://localhost:8001/mcp"   # FastMCP streamable-http endpoint

MAX_ITERATIONS = 5

# ── MCP tools (populated by init_mcp_tools, reused across requests) ───────────
# Keys are the exact tool names registered on the MCP server.
_mcp_client: Any = None          # langchain_mcp_adapters.MultiServerMCPClient instance
_mcp_tools: dict[str, Any] = {}  # {tool_name: LangChain BaseTool}


# ── Tool descriptions (match MCP server tool names exactly) ──────────────────
TOOL_DESCRIPTIONS = """\
- search_documents
    Search the AskAra+ knowledge base: government programs, worker rights,
    health services, flood / emergency relief, social aid, emergency contacts.
    Parameters: {"query": str, "country": str (MY/ID/PH/TH, optional)}

- fetch_gov_portal
    Search government websites for fresh information when the knowledge base
    is insufficient. Pass a descriptive search query, not a URL.
    Parameters: {"query": str, "country": str (optional)}

- profile_match
    Find government programs matching the user's situation and eligibility.
    Parameters: {
      "country": str,
      "situation": str  (worker | business_owner | family | disaster_victim | unemployed | student),
      "need": str       (financial_aid | healthcare | worker_rights | business_support | housing | legal_aid | education)
    }

- simplify
    Simplify complex government text to Grade 5 reading level.
    Parameters: {"text": str, "country": str (optional), "language": str (optional)}
    Tip: omit "text" to automatically use the last retrieved content.

- summarize
    Convert procedural text into numbered step-by-step cards. Use for
    "how to apply / register / claim" questions.
    Parameters: {"text": str, "language": str (optional)}
    Tip: omit "text" to automatically use the last simplified/retrieved content.

- translate
    Translate text between languages.
    Parameters: {"text": str, "source_lang": str, "target_lang": str}

- dialect_adapt
    Adapt text to a regional dialect.
    Valid dialects: kelantan_malay | javanese | waray | kham_mueang
    Parameters: {"text": str, "target_dialect": str}

- FINISH
    You have enough information to give the user a final answer.
    Parameters: {}"""


# ─────────────────────────────────────────────────────────────────────────────
# MCP lifecycle — init / invoke / cleanup
# ─────────────────────────────────────────────────────────────────────────────

async def init_mcp_tools() -> None:
    """
    Connect to the MCP server and cache tool objects.
    Call once from server.py lifespan on startup.
    Safe to call even if the MCP server isn't running yet — failure is logged
    as a warning and the agent falls back to direct imports.
    """
    global _mcp_client, _mcp_tools

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        # langchain-mcp-adapters >=0.1.0 — do NOT use as context manager
        _mcp_client = MultiServerMCPClient({
            "askara": {
                "transport": "streamable_http",
                "url": MCP_SERVER_URL,
            }
        })
        tools_list = await _mcp_client.get_tools()
        _mcp_tools = {t.name: t for t in tools_list}

        logger.info(
            "MCP server connected at %s — %d tools loaded: %s",
            MCP_SERVER_URL,
            len(_mcp_tools),
            list(_mcp_tools.keys()),
        )
    except Exception as exc:
        logger.warning(
            "MCP server not available (%s) — agent will use direct imports as fallback.",
            exc,
        )
        _mcp_client = None
        _mcp_tools = {}


async def _mcp_invoke(tool_name: str, params: dict) -> str:
    """
    Call a tool via MCP server if connected; otherwise fall back to the
    direct Python import.  Returns the result as a JSON string.
    """
    # ── Try MCP ──────────────────────────────────────────────────────────────
    if _mcp_tools and tool_name in _mcp_tools:
        try:
            result = await _mcp_tools[tool_name].ainvoke(params)
            return str(result)
        except Exception as exc:
            logger.warning(
                "MCP tool '%s' failed: %s — falling back to direct import",
                tool_name, exc,
            )

    # ── Direct import fallback ────────────────────────────────────────────────
    return await _direct_invoke(tool_name, params)


async def _direct_invoke(tool_name: str, params: dict) -> str:
    """Call a tool directly via the Python implementation (no MCP)."""
    try:
        if tool_name == "search_documents":
            return _search_documents(
                query=params.get("query", ""),
                country=params.get("country", ""),
                topic=params.get("topic", ""),
            )
        if tool_name == "fetch_gov_portal":
            return await _fetch_gov_portal(
                url=params.get("query", params.get("url", "")),
                country=params.get("country", ""),
            )
        if tool_name == "profile_match":
            return await _profile_match(
                country=params.get("country", ""),
                situation=params.get("situation", ""),
                need=params.get("need", ""),
            )
        if tool_name == "simplify":
            return await _simplify_text(
                text=params.get("text", ""),
                target_grade_level=5,
                country=params.get("country", ""),
                language=params.get("language", ""),
            )
        if tool_name == "summarize":
            return await _summarize_text(
                text=params.get("text", ""),
                format="step_cards",
                language=params.get("language", "en"),
                max_steps=5,
            )
        if tool_name == "translate":
            return await _translate_text(
                text=params.get("text", ""),
                source_lang=params.get("source_lang", "en"),
                target_lang=params.get("target_lang", "en"),
            )
        if tool_name == "dialect_adapt":
            return await _dialect_adapt(
                text=params.get("text", ""),
                target_dialect=params.get("target_dialect", ""),
            )
        return json.dumps({"status": "error", "message": f"Unknown tool: {tool_name}"})

    except Exception as exc:
        logger.error("Direct invoke '%s' failed: %s", tool_name, exc)
        return json.dumps({"status": "error", "message": str(exc)})


# ─────────────────────────────────────────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────────────────────────────────────────

class AraState(TypedDict):
    # Input (read-only after initialisation)
    user_message: str
    country: str
    language: str
    history: list[dict]

    # Context (set by detect_context_node)
    detected_lang: str
    detected_country: str
    detected_dialect: str

    # ReAct loop control
    scratchpad: str
    iterations: int
    last_thought: str
    next_action: str        # MCP tool name or "FINISH"
    next_action_input: dict

    # Accumulated output (each node writes the FULL list, not just a delta)
    tool_calls_made: list[str]
    sources: list[dict]
    structured: Optional[dict]

    # Content pipeline
    last_text_content: str   # most recent text output — auto-fed to next tool
    last_search_text: str    # raw (unsimplified) search/portal text — kept for summarize context
    source_tier: str         # "knowledge_base" | "web"

    # Mismatch recovery (set by tool_executor when agency mismatch detected)
    forced_next_action: str   # non-empty → react_agent_node bypasses LLM reasoning
    forced_next_params: dict  # params to pass to the forced action

    # Query intent (set by detect_context_node — drives forced simplify/summarize pipeline)
    query_intent: str  # "procedural" | "factual"

    # Query intent (set by detect_context_node — drives forced simplify/summarize pipeline)
    query_intent: str  # "procedural" | "factual"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_json_loads(data: Any) -> dict:
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {"status": "success", "results": data}
    if isinstance(data, str):
        cleaned = re.sub(r"```json\s*|```", "", data).strip()
        try:
            parsed = json.loads(cleaned)
        except Exception:
            return {"status": "error", "results": []}
        if isinstance(parsed, list):
            return {"status": "success", "results": parsed}
        return parsed if isinstance(parsed, dict) else {"status": "error", "results": []}
    return {"status": "error", "results": []}


def _try_parse_structured(data: Any) -> dict | None:
    d = _safe_json_loads(data) if not isinstance(data, dict) else data
    if isinstance(d, dict):
        if d.get("type") == "step_cards" and d.get("cards"):
            return d
        if d.get("type") == "recommendations" and d.get("items"):
            return d
    return None


def _has_good_results(result: dict) -> bool:
    if result.get("status") in ("no_results", "low_confidence", "error"):
        return False
    return len(result.get("results", [])) > 0


def _get_text_from_search(result: dict) -> str:
    return "\n\n".join(r.get("text", "") for r in result.get("results", []) if r.get("text"))


def _normalise_url(raw: object) -> str:
    """Coerce a URL value to a plain string.

    ChromaDB metadata can occasionally deserialise a URL field as a list
    (e.g. ``["https://…"]``) instead of a bare string.  Always return a
    single ``str`` so downstream set/hash operations never crash.
    """
    if isinstance(raw, list):
        # Take the first non-empty element, or fall back to ""
        return next((str(u) for u in raw if u), "")
    return str(raw) if raw else ""


def _extract_sources_from_search(result: dict) -> list[dict]:
    sources = []
    for r in result.get("results", []):
        src = r.get("source", {})
        if src:
            entry = {
                "title": src.get("document_title", ""),
                "url": _normalise_url(src.get("document_url", "")),
                "source_agency": src.get("source_agency", ""),
                "country": src.get("country", ""),
                "relevance": r.get("similarity", 0),
            }
            if entry["title"] and entry not in sources:
                sources.append(entry)
    return sources


def _extract_sources_from_portal(result: dict) -> list[dict]:
    sources = []
    for r in result.get("results", []):
        entry = {
            "title": r.get("title", ""),
            "url": _normalise_url(r.get("url", "")),
            "source_agency": "",
            "country": result.get("country", ""),
        }
        if entry["title"] and entry not in sources:
            sources.append(entry)
    return sources


URL_PATTERN = re.compile(r'https?://[^\s)\]>"\']+')


def _collect_allowed_urls(sources: list[dict]) -> set[str]:
    """Return a set of URL strings — safe even if a url field slipped through
    as a list (extra guard on top of the _normalise_url call at extraction)."""
    urls: set[str] = set()
    for s in sources:
        raw = s.get("url", "")
        url = _normalise_url(raw)
        if url:
            urls.add(url)
    return urls


def _strip_hallucinated_urls(text: str, allowed_urls: set[str]) -> str:
    if not allowed_urls:
        return URL_PATTERN.sub("[link removed]", text)

    def _replace(m: re.Match) -> str:
        url = m.group(0).rstrip(".,;:!?)")
        for a in allowed_urls:
            if url.startswith(a) or a.startswith(url):
                return m.group(0)
        return "[link removed]"

    return URL_PATTERN.sub(_replace, text)


def _extract_profiling_data(text: str) -> dict | None:
    country_m = re.search(r"country[:\s]+([A-Z]{2})", text, re.IGNORECASE)
    if not country_m:
        country_m = re.search(r"\(([A-Z]{2})\)", text)
    if not country_m:
        return None

    SITUATION_MAP = {
        "worker": "worker", "business owner": "business_owner",
        "business": "business_owner", "family": "family", "resident": "family",
        "disaster": "disaster_victim", "disaster affected": "disaster_victim",
        "flood": "disaster_victim", "unemployed": "unemployed", "student": "student",
    }
    NEED_MAP = {
        "financial aid": "financial_aid", "financial": "financial_aid",
        "healthcare": "healthcare", "health": "healthcare", "medical": "healthcare",
        "worker rights": "worker_rights", "employment": "worker_rights",
        "legal rights": "legal_aid", "legal aid": "legal_aid", "legal": "legal_aid",
        "business support": "business_support", "housing": "housing",
        "education": "education",
    }

    tl = text.lower()
    sit_m = re.search(
        r"situation[:\s]+(worker|business_owner|family|disaster_victim|unemployed|student)",
        text, re.IGNORECASE,
    )
    found_sit = sit_m.group(1).lower() if sit_m else next(
        (v for k, v in SITUATION_MAP.items() if k in tl), None
    )
    if not found_sit:
        return None

    need_m = re.search(
        r"need[:\s]+(financial_aid|healthcare|worker_rights|business_support|housing|legal_aid|education)",
        text, re.IGNORECASE,
    )
    found_need = need_m.group(1).lower() if need_m else next(
        (v for k, v in NEED_MAP.items() if k in tl), None
    )
    if not found_need:
        return None

    return {
        "country": country_m.group(1).upper(),
        "situation": found_sit,
        "need": found_need,
    }


# ── Search query cleaner (improves embedding retrieval quality) ───────────────
_NOISE = re.compile(
    r"\b("
    r"how\s+(?:to|do\s+I|can\s+I|should\s+I)"
    r"|what\s+(?:is|are|do\s+I\s+need)"
    r"|where\s+(?:to|do\s+I|can\s+I)"
    r"|steps?\s+(?:to|for)"
    r"|cara\s+(?:nak|untuk|memohon|mendaftar|daftar|tuntut|buat|dapatkan)"
    r"|macam\s*mana\s*(?:nak|nok|untuk)?"
    r"|bagaimana\s*(?:cara)?"
    r"|nak\s+(?:mohon|daftar|claim|tuntut|buat)"
    r"|nok\s+(?:daftar|mohon|buat|dapat)"
    r"|paano\s*(?:mag|po)?"
    r"|tolong|please|sila|help\s+me"
    r"|I\s+(?:want|need)\s+to"
    r"|saya\s+(?:nak|mau|ingin)"
    r")\b",
    re.IGNORECASE,
)
_FILLER = re.compile(
    r"\b(the|a|an|for|to|in|at|on|my|me|I|di|ke|dari|yang|dan|atau|saya|aku|po|ko|ka|ba)\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Agency Mismatch Detection
# Maps agency identifiers → keywords that should appear in query OR source titles
# ─────────────────────────────────────────────────────────────────────────────

AGENCY_KEYWORD_MAP: dict[str, list[str]] = {
    # ── MALAYSIA ─────────────────────────────────────────────────────────────
    "KWSP/EPF":    ["kwsp", "epf", "employees provident fund", "provident fund",
                    "i-akaun", "iakaun", "kwsp i-akaun"],
    "SOCSO/PERKESO": ["perkeso", "socso", "assist portal", "employment injury",
                      "social security malaysia", "eis ", "jkk", "jkm perkeso"],
    "LHDN":        ["lhdn", "irb", "inland revenue", "ezhasil", "income tax",
                    "e-filing", "cukai pendapatan", "lembaga hasil"],
    "PTPTN":       ["ptptn", "pinjaman pelajaran", "student loan ptptn"],
    "HRD_CORP":    ["hrd corp", "hrdf", "pembangunan sumber manusia",
                    "human resource development fund"],
    "JKM":         ["jkm", "kebajikan masyarakat", "social welfare malaysia",
                    "jabatan kebajikan"],
    "BSH/BRIM":    ["brim", "bsh", "bantuan sara hidup", "cost of living aid"],
    "MARA":        ["mara", "majlis amanah rakyat"],
    "JPA":         ["jpa", "jabatan perkhidmatan awam", "public service department"],
    "FOMEMA":      ["fomema", "foreign workers medical"],
    "MySejahtera": ["mysejahtera", "mysj"],
    # ── INDONESIA ────────────────────────────────────────────────────────────
    "BPJS_Kesehatan":       ["bpjs kesehatan", "bpjs health", "jaminan kesehatan",
                             "jkn", "kartu indonesia sehat"],
    "BPJS_Ketenagakerjaan": ["bpjs ketenagakerjaan", "bpjs employment",
                             "jamsostek", "bpjs tk"],
    "Kemnaker":             ["disnaker", "dinas tenaga kerja", "kemenaker",
                             "ministry of manpower indonesia"],
    "Disnakertrans":        ["disnakertrans", "transmigration"],
    # ── PHILIPPINES ──────────────────────────────────────────────────────────
    "PhilHealth":  ["philhealth", "philippine health insurance", "nhic"],
    "SSS":         ["sss", "social security system philippines"],
    "Pag-IBIG":    ["pag-ibig", "pagibig", "hdmf", "home development mutual fund"],
    "DOLE":        ["dole", "department of labor philippines"],
    "DTI":         ["dti", "department of trade philippines", "negosyo center"],
    "OWWA":        ["owwa", "overseas workers welfare"],
    # ── THAILAND ─────────────────────────────────────────────────────────────
    "SSO_Thailand":  ["sso thailand", "social security office thailand",
                      "ประกันสังคม", "prakan sangkhom"],
    "NHSO_Thailand": ["nhso", "สปสช", "universal coverage scheme thailand",
                      "บัตรทอง", "gold card thailand"],
    "DBD_Thailand":  ["dbd thailand", "department of business development thailand",
                      "กรมพัฒนาธุรกิจการค้า"],
}


def _detect_agency_mismatch(query: str, result: dict) -> dict | None:
    """
    Deterministically detect when the user's query references a specific agency
    but the retrieved source documents are from a *different* agency.

    Returns a mismatch info dict if a mismatch is detected, None otherwise.
    A None return means: either no known agency in the query, or sources match.
    """
    query_lower = query.lower()

    # Step 1: Which agency is the user asking about?
    queried_agency: str | None = None
    for agency, keywords in AGENCY_KEYWORD_MAP.items():
        if any(kw in query_lower for kw in keywords):
            queried_agency = agency
            break

    if not queried_agency:
        return None  # Query doesn't mention a known agency — can't detect mismatch

    # Step 2: What agencies appear in the returned source document titles?
    results = result.get("results", [])
    if not results:
        return None  # No results — let normal fallback logic handle it

    source_text = " ".join(
        r.get("source", {}).get("document_title", "").lower()
        + " " + r.get("source", {}).get("source_agency", "").lower()
        for r in results
    )

    # Step 3: Do any of the queried agency's keywords appear in the source titles?
    queried_keywords = AGENCY_KEYWORD_MAP[queried_agency]
    if any(kw in source_text for kw in queried_keywords):
        return None  # Sources match the queried agency — all good

    # Step 4: Which agencies ARE in the sources?
    found_source_agencies = [
        agency for agency, keywords in AGENCY_KEYWORD_MAP.items()
        if any(kw in source_text for kw in keywords)
    ]

    return {
        "queried_agency": queried_agency,
        "source_agencies": found_source_agencies or ["unknown"],
        "source_titles": [
            r.get("source", {}).get("document_title", "?") for r in results[:3]
        ],
    }


def _build_search_observation(result: dict, mismatch: dict | None) -> str:
    """
    Build a human-readable scratchpad observation for search_documents results.
    Much clearer than raw JSON — lets the ReAct LLM make correct next-step decisions.
    """
    status = result.get("status", "unknown")
    results = result.get("results", [])

    if status in ("no_results", "low_confidence", "error") or not results:
        return (
            f"[search_documents] status={status}: {result.get('message', 'No results found.')} "
            f"→ Use fetch_gov_portal to search the web instead."
        )

    lines = [f"[search_documents] Found {len(results)} chunks:"]
    for i, r in enumerate(results):
        src = r.get("source", {})
        title = src.get("document_title", "Unknown")
        agency = src.get("source_agency", "")
        country = src.get("country", "")
        sim = r.get("similarity", 0)
        band = "HIGH" if sim >= 0.75 else "MED" if sim >= 0.60 else "LOW"
        label = f"[{agency} / {country}]" if agency else f"[{country}]"
        lines.append(f"  [{i}] sim={sim:.3f} ({band}) | {title} {label}")

    if mismatch:
        queried = mismatch["queried_agency"]
        sources_str = ", ".join(mismatch["source_agencies"])
        lines += [
            "",
            f"⚠ AGENCY MISMATCH DETECTED:",
            f"  User asked about: {queried}",
            f"  Sources returned: {sources_str}",
            f"  These results are NOT relevant to the user's question.",
            f"→ REQUIRED NEXT STEP: call fetch_gov_portal to get correct information.",
        ]
    else:
        lines.append("✓ Sources appear relevant to the query.")

    return "\n".join(lines)


def _clean_query(query: str) -> str:
    c = _NOISE.sub("", query)
    c = re.sub(r"[?!.,;:\"'()]+", " ", c)
    stripped = re.sub(r"\s+", " ", _FILLER.sub("", c)).strip()
    return stripped if len(stripped) >= 3 else c.strip() or query.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Tool Wrappers
# Pre-process params (fill defaults from state), then call _mcp_invoke.
# Returns (result_dict, new_sources, text_content).
# ─────────────────────────────────────────────────────────────────────────────

async def _run_search(params: dict, state: AraState) -> tuple[dict, list[dict], str]:
    query = _clean_query(params.get("query", state["user_message"]))
    country = params.get("country", state["detected_country"])

    raw = await _mcp_invoke("search_documents", {"query": query, "country": country})
    result = _safe_json_loads(raw)

    # Retry without country filter on empty results
    if not _has_good_results(result) and country:
        logger.info("Search: empty with country=%s — retrying without country filter", country)
        raw = await _mcp_invoke("search_documents", {"query": query, "country": ""})
        result = _safe_json_loads(raw)

    return result, _extract_sources_from_search(result), _get_text_from_search(result)


async def _run_portal(params: dict, state: AraState) -> tuple[dict, list[dict], str]:
    query = params.get("query", params.get("url", state["user_message"]))
    country = params.get("country", state["detected_country"])

    raw = await _mcp_invoke("fetch_gov_portal", {"query": query, "country": country})
    result = _safe_json_loads(raw)
    return result, _extract_sources_from_portal(result), result.get("content", "")


async def _run_profile_match(params: dict, state: AraState) -> tuple[dict, list[dict], str]:
    country = params.get("country", state["detected_country"])
    situation = params.get("situation", "")
    need = params.get("need", "")

    # Supplement from user message when the agent omits fields
    if not (situation and need):
        profile = _extract_profiling_data(state["user_message"])
        if profile:
            country = country or profile.get("country", "")
            situation = situation or profile.get("situation", "")
            need = need or profile.get("need", "")

    raw = await _mcp_invoke("profile_match", {
        "country": country, "situation": situation, "need": need,
    })
    return _safe_json_loads(raw), [], ""


async def _run_simplify(params: dict, state: AraState) -> tuple[dict, list[dict], str]:
    text = params.get("text") or state.get("last_text_content", "")
    if not text:
        return {"status": "error", "message": "No text available to simplify"}, [], ""

    raw = await _mcp_invoke("simplify", {
        "text": text[:6000],           # increased from 3000 — captures all retrieved chunks
        "target_grade_level": 5,
        "country": state["detected_country"],
        "language": state["detected_lang"],
    })
    result = _safe_json_loads(raw)
    return result, [], result.get("simplified_text", text)


async def _run_summarize(params: dict, state: AraState) -> tuple[dict, list[dict], str]:
    simplified_text = params.get("text") or state.get("last_text_content", "")
    raw_search_text = state.get("last_search_text", "")
    user_question = state.get("user_message", "")

    if not simplified_text and not raw_search_text:
        return {"status": "error", "message": "No text available to summarize"}, [], ""

    # Build rich context: user question + raw search text + simplified text.
    # Giving the LLM all three lets it generate a COMPLETE procedural guide
    # even when the knowledge base only returned eligibility/exemption clauses.
    context_parts = []
    if user_question:
        context_parts.append("USER QUESTION: " + user_question)
    if raw_search_text:
        context_parts.append("RETRIEVED INFORMATION:\n" + raw_search_text[:4000])
    if simplified_text and simplified_text != raw_search_text:
        context_parts.append("SIMPLIFIED SUMMARY:\n" + simplified_text[:2000])

    rich_text = "\n\n---\n\n".join(context_parts)

    raw = await _mcp_invoke("summarize", {
        "text": rich_text,
        "format": "step_cards",
        "language": state["detected_lang"],
        "max_steps": 5,
    })
    return _safe_json_loads(raw), [], ""


async def _run_translate(params: dict, state: AraState) -> tuple[dict, list[dict], str]:
    text = params.get("text") or state.get("last_text_content", "")
    raw = await _mcp_invoke("translate", {
        "text": text[:2000],
        "source_lang": params.get("source_lang", "en"),
        "target_lang": params.get("target_lang", state["detected_lang"]),
    })
    result = _safe_json_loads(raw)
    return result, [], result.get("translated_text", "")


async def _run_dialect(params: dict, state: AraState) -> tuple[dict, list[dict], str]:
    text = params.get("text") or state.get("last_text_content", "")
    raw = await _mcp_invoke("dialect_adapt", {
        "text": text[:2000],
        "target_dialect": params.get("target_dialect", state["detected_dialect"]),
    })
    result = _safe_json_loads(raw)
    return result, [], result.get("adapted_text", "")


TOOL_RUNNERS: dict[str, Any] = {
    "search_documents": _run_search,
    "fetch_gov_portal": _run_portal,
    "profile_match": _run_profile_match,
    "simplify": _run_simplify,
    "summarize": _run_summarize,
    "translate": _run_translate,
    "dialect_adapt": _run_dialect,
}

_VALID_ACTIONS = frozenset(TOOL_RUNNERS.keys()) | {"FINISH"}


# ─────────────────────────────────────────────────────────────────────────────
# ReAct Prompt Builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_react_prompt(state: AraState) -> str:
    history_lines = []
    for turn in (state.get("history") or [])[-4:]:
        role = turn.get("role", "")
        content = str(turn.get("content", ""))[:200]
        history_lines.append(f"{role.capitalize()}: {content}")
    history_text = "\n".join(history_lines) or "(none)"

    scratchpad = state.get("scratchpad", "").strip() or "(first step — nothing done yet)"
    remaining = MAX_ITERATIONS - state.get("iterations", 0)

    last_text = state.get("last_text_content", "")
    chain_hint = (
        f'\nLast retrieved text (for tool chaining): "{last_text[:200]}'
        f'{"..." if len(last_text) > 200 else ""}"'
        if last_text else ""
    )

    mcp_status = (
        f"✓ MCP server connected ({len(_mcp_tools)} tools)"
        if _mcp_tools else "⚠ MCP offline — using direct imports"
    )

    # Build a clear "already used / still available" split
    used_tools = state.get("tool_calls_made", [])
    used_set = set(used_tools)
    all_tools = list(TOOL_RUNNERS.keys())
    available_tools = [t for t in all_tools if t not in used_set]

    used_block = (
        f"ALREADY CALLED (FORBIDDEN — do NOT call again): {', '.join(used_set)}"
        if used_set else "ALREADY CALLED: (none yet)"
    )
    available_block = (
        f"STILL AVAILABLE: {', '.join(available_tools)}"
        if available_tools else "STILL AVAILABLE: (none — you MUST use FINISH)"
    )

    return f"""\
You are Ara, an AI assistant for AskAra+ helping ASEAN migrant workers and \
vulnerable populations access government services.

## Available Tools
{TOOL_DESCRIPTIONS}

## STRICT Response Format
Respond using EXACTLY this structure. Nothing before "Thought:", nothing after "Action Input:".

Thought: <your step-by-step reasoning>
Action: <exact tool name from the list above, or FINISH>
Action Input: <a valid JSON object with tool parameters>

## ⚠ ONE-TIME TOOL RULE — CRITICAL
Each tool may be called AT MOST ONCE per user message. This is a hard rule.
{used_block}
{available_block}
If you want to call a FORBIDDEN tool again → use FINISH instead.
If STILL AVAILABLE is empty → use FINISH immediately.

## Decision Rules
1. Greetings / social / "thank you" → FINISH immediately
2. Any question about a program, right, or procedure → search_documents (if not yet used), then FINISH
   NOTE: simplify and summarize are triggered AUTOMATICALLY by the pipeline — you do NOT need to call them
3. "What programs can I get / am I eligible for" → profile_match → FINISH
4. If search_documents returns empty/poor results AND fetch_gov_portal not yet used → fetch_gov_portal
5. After summarize or profile_match → FINISH immediately, no more tools
6. Omit "text" in simplify / summarize — it auto-uses the last retrieved content
7. NEVER invent program names, benefit amounts, or URLs
8. NEVER call simplify or summarize yourself — the pipeline calls them automatically after search_documents

## Session Context
Language     : {state.get("detected_lang", "en")}
Country      : {state.get("detected_country", "unknown")}
Dialect      : {state.get("detected_dialect", "standard")}
Query intent : {state.get("query_intent", "factual")} (auto-detected — pipeline enforced)
Tool calls remaining: {remaining}
Backend      : {mcp_status}{chain_hint}

## Conversation History
{history_text}

## Reasoning Progress
{scratchpad}

User message: {state["user_message"]}

Respond now with Thought / Action / Action Input:\
"""


# ─────────────────────────────────────────────────────────────────────────────
# ReAct Response Parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_react_response(text: str) -> tuple[str, str, dict]:
    """
    Parse LLM output → (thought, action, action_input).
    Returns ("", "FINISH", {}) on any failure — fail-safe.
    """
    thought = ""
    action = "FINISH"
    action_input: dict = {}

    m = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", text, re.DOTALL | re.IGNORECASE)
    if m:
        thought = m.group(1).strip()

    m = re.search(r"Action:\s*(\S+)", text, re.IGNORECASE)
    if m:
        action = m.group(1).strip().rstrip(".,")

    m = re.search(r"Action\s+Input:\s*(\{.*?\})\s*$", text, re.DOTALL | re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
        for candidate in (raw, raw.replace("'", '"')):
            try:
                action_input = json.loads(candidate)
                break
            except json.JSONDecodeError:
                pass
        else:
            logger.warning("Could not parse Action Input JSON: %s", raw[:120])

    # Normalise action name (case-insensitive fuzzy match)
    if action not in _VALID_ACTIONS:
        lower = action.lower()
        matched = next((a for a in _VALID_ACTIONS if a.lower() == lower), None)
        if matched:
            action = matched
        else:
            logger.warning("Unrecognised action '%s' — defaulting to FINISH", action)
            action = "FINISH"

    logger.info("ReAct parsed → action=%s  input=%s", action, str(action_input)[:120])
    return thought, action, action_input


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph Nodes
# ─────────────────────────────────────────────────────────────────────────────

# Keywords that signal the user wants a step-by-step procedure (TYPE B).
# When any appear, the pipeline DETERMINISTICALLY runs:
#   search_documents → simplify → summarize → FINISH
# For TYPE A (factual) queries: search_documents → simplify → FINISH
_PROCEDURAL_SIGNALS = [
    # English
    "how to ", "how do i ", "how can i ", "how do you ", "how do we ",
    "what steps", "what do i need to", "what do i need for",
    "where do i go", "where to go", "where to apply", "where to register",
    "where can i go", "step by step", "procedure", "process to ",
    # Action verbs mapping to registration / application workflows
    "register", "apply ", "applying ", "claim ", "claiming ", "enrol", "enroll",
    "renew ", "renewing ", "submit ", "open a business", "start a business",
    "start my business", "set up a business", "get a permit", "obtain a ",
    # Malay / Indonesian
    "macam mana", "cara nak", "cara untuk", "cara mendapat", "bagaimana cara",
    "nak daftar", "nak mohon", "nak apply", "mendaftar", "permohonan",
    "mohon untuk", "cara memohon", "langkah-langkah", "prosedur",
    "di mana boleh", "ke mana pergi",
    # Filipino / Tagalog
    "paano ", "pano mag", "kung paano", "mag-apply", "mag-register",
    "magrerehistro", "paano kumuha", "saan pwede",
    # Thai
    "วิธี", "ขั้นตอน", "ทำอย่างไร", "ลงทะเบียน", "สมัคร",
]


def _detect_query_intent(message: str) -> str:
    """
    Return "procedural" if the message looks like a step-by-step / how-to query,
    otherwise return "factual".  Keyword-based — deterministic, no LLM needed.
    """
    msg_lower = message.lower()
    if any(signal in msg_lower for signal in _PROCEDURAL_SIGNALS):
        return "procedural"
    return "factual"


def detect_context_node(state: AraState) -> dict:
    """
    Node 1 — Detect language / dialect / country / query intent (sync, no LLM, no MCP).
    Runs directly via the Python import — fast and dependency-free.
    """
    message = state["user_message"]
    country_hint = state.get("country", "")
    language_hint = state.get("language", "")

    try:
        lang_data = json.loads(_detect_language(text=message))
        detected_lang = lang_data.get("primary_lang", "en")
        detected_dialect = lang_data.get("dialect", "standard")
        detected_country = lang_data.get("country_hint", "") or country_hint
    except Exception as exc:
        logger.warning("Language detection failed: %s", exc)
        detected_lang = language_hint or "en"
        detected_dialect = "standard"
        detected_country = country_hint

    query_intent = _detect_query_intent(message)

    logger.info(
        "Context → lang=%s  dialect=%s  country=%s  intent=%s  mcp_tools=%d",
        detected_lang, detected_dialect, detected_country, query_intent, len(_mcp_tools),
    )
    return {
        "detected_lang": detected_lang,
        "detected_dialect": detected_dialect,
        "detected_country": detected_country,
        "query_intent": query_intent,
    }


async def react_agent_node(state: AraState) -> dict:
    """
    Node 2 — LLM reasons about next action via ReAct prompting (non-streaming).
    Hard rule: any tool already in tool_calls_made is blocked — FINISH is forced.
    """
    used_tools: set[str] = set(state.get("tool_calls_made", []))
    available_tools = [t for t in TOOL_RUNNERS if t not in used_tools]

    # ── Forced action from mismatch detection (bypasses LLM entirely) ────────
    forced = state.get("forced_next_action", "")
    if forced and forced not in used_tools and forced in TOOL_RUNNERS:
        forced_params = state.get("forced_next_params", {})
        thought = (
            f"Knowledge base returned documents from a different agency than what "
            f"the user asked about. Switching to {forced} to retrieve correct information."
        )
        logger.info("react_agent_node: forced action '%s' params=%s", forced, forced_params)
        return {
            "last_thought": thought,
            "next_action": forced,
            "next_action_input": forced_params,
            "forced_next_action": "",
            "forced_next_params": {},
            "scratchpad": (
                state.get("scratchpad", "") +
                f"\nThought: {thought}\nAction: {forced}\n"
                f"Action Input: {json.dumps(forced_params)}\n"
            ),
            "iterations": state.get("iterations", 0) + 1,
        }

    # Hard stop: no tools left or max iterations
    if state.get("iterations", 0) >= MAX_ITERATIONS or not available_tools:
        reason = (
            f"Max iterations ({MAX_ITERATIONS}) reached."
            if state.get("iterations", 0) >= MAX_ITERATIONS
            else "All tools have been used once — finishing."
        )
        logger.info("react_agent_node: %s", reason)
        return {
            "last_thought": "I have gathered enough information to answer the user.",
            "next_action": "FINISH",
            "next_action_input": {},
            "iterations": state.get("iterations", 0) + 1,
        }

    try:
        # Pass SYSTEM_PROMPT so tool rules are visible to the reasoning LLM
        llm_response = await call_llm(
            _build_react_prompt(state),
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,
        )
    except LLMError as exc:
        logger.error("LLM error in react_agent_node: %s", exc)
        return {
            "last_thought": "LLM temporarily unavailable — finishing with available info.",
            "next_action": "FINISH",
            "next_action_input": {},
            "iterations": state.get("iterations", 0) + 1,
        }

    thought, action, action_input = _parse_react_response(llm_response)

    # ── Hard guard: block re-use of any already-called tool ──────────────────
    if action != "FINISH" and action in used_tools:
        logger.warning(
            "Agent tried to re-call '%s' (already used) — forcing FINISH. "
            "Used tools: %s",
            action, used_tools,
        )
        thought = (
            f"I already called '{action}'. I have enough information — "
            "I will now give the user the final answer."
        )
        action = "FINISH"
        action_input = {}

    return {
        "last_thought": thought,
        "next_action": action,
        "next_action_input": action_input,
        "scratchpad": (
            state.get("scratchpad", "") +
            f"\nThought: {thought}\nAction: {action}\n"
            f"Action Input: {json.dumps(action_input)}\n"
        ),
        "iterations": state.get("iterations", 0) + 1,
    }


async def tool_executor_node(state: AraState) -> dict:
    """
    Node 3 — Execute the chosen tool via MCP (or direct import fallback).
    Updates: tool_calls_made, sources, structured, last_text_content, source_tier.
    """
    tool_name = state["next_action"]
    params = state.get("next_action_input", {})

    runner = TOOL_RUNNERS.get(tool_name)
    if not runner:
        logger.error("No runner for tool: %s", tool_name)
        obs = json.dumps({"status": "error", "message": f"Unknown tool: {tool_name}"})
        return {
            "scratchpad": state.get("scratchpad", "") + f"Observation: {obs}\n",
            "tool_calls_made": state.get("tool_calls_made", []) + [tool_name],
        }

    try:
        result_dict, new_sources, text_content = await runner(params, state)
    except Exception as exc:
        logger.error("Tool '%s' raised: %s", tool_name, exc, exc_info=True)
        result_dict = {"status": "error", "message": str(exc)}
        new_sources, text_content = [], ""

    obs_str = json.dumps(result_dict)

    # ── Build scratchpad observation ─────────────────────────────────────────
    # search_documents gets a human-readable summary (LLM can read source titles).
    # All other tools get raw JSON, with a generous 1 200-char budget.
    mismatch_info: dict | None = None
    if tool_name == "search_documents":
        mismatch_info = _detect_agency_mismatch(state["user_message"], result_dict)
        obs_line = _build_search_observation(result_dict, mismatch=mismatch_info)
    else:
        obs_line = obs_str[:1200] + ("…" if len(obs_str) > 1200 else "")

    update: dict = {
        "scratchpad": state.get("scratchpad", "") + f"Observation: {obs_line}\n",
        "tool_calls_made": state.get("tool_calls_made", []) + [tool_name],
        "sources": state.get("sources", []) + new_sources,
        # Reset forced fields by default — set below only when needed
        "forced_next_action": "",
        "forced_next_params": {},
    }

    if text_content:
        update["last_text_content"] = text_content
        # Preserve the raw (unsimplified) retrieval text so summarize can use it
        # for richer context even after simplify compresses the content.
        if tool_name in ("search_documents", "fetch_gov_portal"):
            update["last_search_text"] = text_content

    if tool_name == "fetch_gov_portal":
        update["source_tier"] = "web"
    elif tool_name == "search_documents" and not state.get("source_tier"):
        update["source_tier"] = "knowledge_base"

    # ── Agency mismatch → force fetch_gov_portal if not yet used ────────────
    if mismatch_info:
        already_used = set(state.get("tool_calls_made", []) + [tool_name])
        if "fetch_gov_portal" not in already_used:
            queried = mismatch_info["queried_agency"]
            logger.info(
                "Agency mismatch: query mentions %s but sources are %s — "
                "forcing fetch_gov_portal",
                queried, mismatch_info["source_agencies"],
            )
            update["forced_next_action"] = "fetch_gov_portal"
            update["forced_next_params"] = {
                "query": f"{queried} {state['user_message']}",
                "country": state.get("detected_country", ""),
            }

    # ── Deterministic simplify → summarize pipeline ───────────────────────────
    # After search_documents or fetch_gov_portal returns useful text content,
    # ALWAYS force simplify next (so government jargon is never shown raw).
    # After simplify, if the query is procedural, force summarize to produce
    # step cards.  This mirrors the TYPE A / TYPE B chains in the system prompt
    # but makes them deterministic — independent of LLM judgment.
    already_used_pipeline = set(state.get("tool_calls_made", []) + [tool_name])

    if not mismatch_info and tool_name in ("search_documents", "fetch_gov_portal") and text_content:
        intent = state.get("query_intent", "factual")
        if intent == "procedural":
            # Procedural: skip simplify entirely — summarize handles simplification
            # internally with its TIER 1/TIER 2 prompt.  Saves 1 LLM call.
            if "summarize" not in already_used_pipeline:
                logger.info(
                    "Pipeline: %s content (%d chars), intent=procedural → skipping simplify, forcing summarize",
                    tool_name, len(text_content),
                )
                update["forced_next_action"] = "summarize"
                update["forced_next_params"] = {}
        else:
            # Factual: simplify first so the final LLM gets clean input.
            if "simplify" not in already_used_pipeline:
                logger.info(
                    "Pipeline: %s returned content (%d chars), intent=factual → forcing simplify",
                    tool_name, len(text_content),
                )
                update["forced_next_action"] = "simplify"
                update["forced_next_params"] = {}

    elif tool_name == "simplify" and text_content:
        # Factual path post-simplify: no summarize needed — final LLM writes the answer.
        logger.info("Pipeline: simplify done (factual path) → finishing to final LLM answer")

    structured = _try_parse_structured(result_dict)
    if structured:
        update["structured"] = structured

    return update


# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_agent(state: AraState) -> str:
    if state.get("next_action") == "FINISH":
        return END
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return END
    return "tool_executor"


# ─────────────────────────────────────────────────────────────────────────────
# Graph (compiled once at import time, reused across all requests)
# ─────────────────────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(AraState)
    g.add_node("detect_context", detect_context_node)
    g.add_node("react_agent", react_agent_node)
    g.add_node("tool_executor", tool_executor_node)

    g.set_entry_point("detect_context")
    g.add_edge("detect_context", "react_agent")
    g.add_conditional_edges(
        "react_agent",
        _route_after_agent,
        {"tool_executor": "tool_executor", END: END},
    )
    g.add_edge("tool_executor", "react_agent")
    return g.compile()


_COMPILED_GRAPH = None


def _get_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = _build_graph()
        logger.info("LangGraph ReAct agent compiled.")
    return _COMPILED_GRAPH


# ─────────────────────────────────────────────────────────────────────────────
# Final LLM prompt (assembled after graph, drives the streaming answer)
# ─────────────────────────────────────────────────────────────────────────────

def _build_final_prompt(state: dict) -> str:
    lang = state.get("detected_lang", "en")
    user_msg = state.get("user_message", "")
    structured = state.get("structured")
    last_text = state.get("last_text_content", "")
    source_tier = state.get("source_tier", "knowledge_base")

    if structured and structured.get("type") == "step_cards":
        return (
            f"Step-by-step cards have been prepared for the user's question. "
            f"Write ONLY a warm 1–2 sentence introduction in {lang}. "
            f"Do NOT list or repeat the steps — they are already shown as cards.\n\n"
            f"User question: {user_msg}"
        )
    if structured and structured.get("type") == "recommendations":
        total = structured.get("total_matches", len(structured.get("items", [])))
        return (
            f"Found {total} government program recommendations. "
            f"Write ONLY a warm 1–2 sentence introduction in {lang}. "
            f"Do NOT list the programs — they are already shown as cards.\n\n"
            f"User question: {user_msg}"
        )
    if last_text:
        # ── Build source-awareness context ────────────────────────────────────
        # Give the answering LLM the retrieved document titles so it can
        # self-detect a mismatch and refuse to hallucinate an answer.
        sources = state.get("sources", [])
        source_context = ""
        if sources:
            titles = list(dict.fromkeys(  # deduplicated, insertion-ordered
                s.get("title", "") for s in sources if s.get("title")
            ))[:4]
            if titles:
                titles_str = "; ".join(titles)
                source_context = (
                    f"\n\nSOURCE DOCUMENTS RETRIEVED: {titles_str}\n"
                    f"⚠ ACCURACY CHECK — Before writing your answer, verify:\n"
                    f"  • Are the source documents actually about what the user asked: "
                    f"'{user_msg[:70]}'?\n"
                    f"  • If the sources are about a DIFFERENT agency or unrelated topic, "
                    f"do NOT use their content to fabricate an answer.\n"
                    f"  • In that case, honestly tell the user you could not find specific "
                    f"information and suggest they contact the relevant agency directly.\n"
                )

        web_note = (
            "\n\nIMPORTANT: Add a brief note that this info is from a government "
            "website and should be verified with the relevant agency."
            if source_tier == "web" else ""
        )
        return (
            f"Here is the information retrieved to answer the user's question:\n\n"
            f"{last_text[:2000]}"
            f"{source_context}\n\n"
            f"Write a clear, warm, helpful response. Grade 5 reading level. "
            f"Respond in {lang}.{web_note}\n\nUser question: {user_msg}"
        )
    return (
        f"I searched the knowledge base and government portals but could not find "
        f"specific information for this question.\n\n"
        f"Write an empathetic response in {lang}: acknowledge the limitation briefly, "
        f"suggest contacting the relevant government agency or helpline. "
        f"Keep it warm and concise.\n\nUser question: {user_msg}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API — Streaming
# ─────────────────────────────────────────────────────────────────────────────

async def run_agent_streaming(
    message: str,
    *,
    country: str | None = None,
    language: str | None = None,
    history: list[dict] | None = None,
    image_base64: str | None = None,
    image_media_type: str = "image/jpeg",
) -> AsyncGenerator[dict, None]:
    """
    Run the LangGraph ReAct agent and yield typed WebSocket events.

    When image_base64 is provided the normal ReAct graph is bypassed entirely.
    The image + optional text are sent directly to the SEA-LION vision model
    (Gemma-SEA-LION-v4-27B-IT) which streams back its analysis.

    Phase 1 — graph.astream(stream_mode="updates"):
        Iterates node by node; emits tool_start / tool_end / reasoning events.
    Phase 2 — call_llm_streaming:
        Streams the final answer once the graph completes.
    """
    # ── Vision fast-path — bypass ReAct graph entirely ──────────────────────
    if image_base64:
        logger.info(
            "Vision fast-path: image_media_type=%s  prompt_len=%d",
            image_media_type,
            len(message),
        )
        full_response = ""
        try:
            async for token in call_llm_vision_streaming(
                image_base64,
                message,
                system_prompt=SYSTEM_PROMPT,
                image_media_type=image_media_type,
            ):
                full_response += token
                yield {"type": "token", "content": token}

            yield {"type": "done", "content": full_response}

        except Exception as exc:
            logger.error("Vision fast-path error: %s", exc, exc_info=True)
            yield {
                "type": "error",
                "content": "Sorry, I couldn't analyse the image. Please try again.",
            }
        return  # <-- do not fall through to the ReAct graph

    # ── Normal text-only path ────────────────────────────────────────────────
    full_response = ""

    initial_state: AraState = {
        "user_message": message,
        "country": country or "",
        "language": language or "",
        "history": history or [],
        "detected_lang": language or "en",
        "detected_country": country or "",
        "detected_dialect": "standard",
        "scratchpad": "",
        "iterations": 0,
        "last_thought": "",
        "next_action": "",
        "next_action_input": {},
        "tool_calls_made": [],
        "sources": [],
        "structured": None,
        "last_text_content": "",
        "last_search_text": "",
        "source_tier": "knowledge_base",
        "forced_next_action": "",
        "forced_next_params": {},
        "query_intent": "factual",  # overwritten by detect_context_node
    }

    final_state: dict = dict(initial_state)
    structured_emitted = False

    try:
        graph = _get_graph()

        # ── Phase 1: graph streaming ─────────────────────────────────────────
        async for chunk in graph.astream(initial_state, stream_mode="updates"):
            for node_name, state_update in chunk.items():
                final_state.update(state_update)

                if node_name == "react_agent":
                    thought = state_update.get("last_thought", "")
                    next_action = state_update.get("next_action", "")

                    if thought:
                        yield {"type": "reasoning", "content": thought}
                    if next_action and next_action != "FINISH":
                        yield {"type": "tool_start", "content": next_action}

                elif node_name == "tool_executor":
                    calls = state_update.get("tool_calls_made", [])
                    if calls:
                        yield {"type": "tool_end", "content": calls[-1]}

                    new_structured = state_update.get("structured")
                    if new_structured and not structured_emitted:
                        structured_emitted = True
                        yield {"type": "structured", "content": new_structured}

        # ── Emit sources ─────────────────────────────────────────────────────
        all_sources: list[dict] = final_state.get("sources", [])
        if all_sources:
            yield {"type": "sources", "content": all_sources}

        # ── Phase 2: stream final answer (or use step_cards summary) ──────────
        # When step_cards are ready, the "summary" field already IS the warm
        # 1-2 sentence intro the LLM would generate.  Re-using it saves an
        # entire LLM call (the biggest single win for the rate limit budget).
        final_structured = final_state.get("structured")
        if final_structured and final_structured.get("type") == "step_cards":
            intro = final_structured.get("summary", "").strip()
            if not intro:
                lang = final_state.get("detected_lang", "en")
                intro = f"Here are the steps to help you with your request."
            logger.info(
                "Phase 2 skipped — using step_cards summary as intro (%d chars). "
                "Saved 1 LLM call.",
                len(intro),
            )
            full_response = intro
            yield {"type": "token", "content": intro}
        elif final_structured and final_structured.get("type") == "recommendations":
            # Same optimisation for profile_match recommendation cards
            lang = final_state.get("detected_lang", "en")
            total = final_structured.get("total_matches", len(final_structured.get("items", [])))
            intro = f"I found {total} government programs that may help you."
            logger.info("Phase 2 skipped — using recommendations intro. Saved 1 LLM call.")
            full_response = intro
            yield {"type": "token", "content": intro}
        else:
            # Normal path: no structured cards — stream the final LLM answer
            async for token in call_llm_streaming(
                _build_final_prompt(final_state),
                system_prompt=SYSTEM_PROMPT,
                history=history,
            ):
                full_response += token
                yield {"type": "token", "content": token}

            full_response = _strip_hallucinated_urls(
                full_response, _collect_allowed_urls(all_sources)
            )

        yield {"type": "done", "content": full_response}

    except Exception as exc:
        logger.error("Agent pipeline error: %s", exc, exc_info=True)
        yield {"type": "error", "content": "I'm sorry, I encountered an issue. Please try again."}


# ─────────────────────────────────────────────────────────────────────────────
# Public API — Non-streaming
# ─────────────────────────────────────────────────────────────────────────────

async def run_agent(
    message: str,
    *,
    country: str | None = None,
    language: str | None = None,
    history: list[dict] | None = None,
    image_base64: str | None = None,
    image_media_type: str = "image/jpeg",
) -> dict:
    """Non-streaming wrapper. Returns {reply, sources, tool_calls, structured}."""
    tool_calls: list[str] = []
    sources: list[dict] = []
    structured = None
    full_response = ""

    async for event in run_agent_streaming(
        message,
        country=country,
        language=language,
        history=history,
        image_base64=image_base64,
        image_media_type=image_media_type,
    ):
        etype = event.get("type", "")
        if etype == "token":
            full_response += event.get("content", "")
        elif etype == "tool_start":
            tool_calls.append(event.get("content", ""))
        elif etype == "structured":
            structured = event.get("content")
        elif etype == "sources":
            sources = event.get("content", [])
        elif etype == "done":
            full_response = event.get("content", full_response)
        elif etype == "error":
            full_response = event.get("content", "Sorry, something went wrong.")

    return {
        "reply": full_response,
        "sources": sources,
        "tool_calls": tool_calls,
        "structured": structured,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle (called by server.py lifespan)
# ─────────────────────────────────────────────────────────────────────────────

async def cleanup_agent() -> None:
    """Release the MCP client on app shutdown."""
    global _mcp_client, _mcp_tools
    if _mcp_client is not None:
        _mcp_client = None
        _mcp_tools = {}
        logger.info("MCP client released.")
    logger.info("LangGraph ReAct agent shutdown complete.")