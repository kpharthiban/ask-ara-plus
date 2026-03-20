"""
AskAra+ — Deterministic Agent Pipeline
-----------------------------------------
Fixed tool-call order per query type. SEA-LION is used ONLY for
text generation — never for deciding which tools to call.

Tools are imported directly from tools/*.py (no MCP at runtime).

Architecture:
  1. Classify query → TYPE A/B/C/D/E
  2. Execute fixed tool chain for that type
  3. Feed tool results into LLM for final response
  4. Yield streaming events to the WebSocket

Usage:
    from agent import run_agent, run_agent_streaming
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncGenerator
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("askara.agent")

# ---------------------------------------------------------------------------
# Tool imports — direct, no MCP
# ---------------------------------------------------------------------------

from tools.search import search_documents as _search_documents
from tools.language import detect_language as _detect_language
from tools.simplify import simplify_text as _simplify_text
from tools.translate import translate_text as _translate_text
from tools.summarize import summarize_text as _summarize_text
from tools.dialect import dialect_adapt as _dialect_adapt
from tools.portal import fetch_gov_portal as _fetch_gov_portal
from tools.profiler import profile_match as _profile_match
# scanner is called from server.py directly via /api/scan endpoint

# ---------------------------------------------------------------------------
# LLM client (for final response generation)
# ---------------------------------------------------------------------------

from llm_client import call_llm, call_llm_streaming, LLMError

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_prompt_file = _PROMPTS_DIR / "system_prompt.txt"

# Also try the same directory as agent.py (flat layout fallback)
if not _prompt_file.exists():
    _prompt_file = Path(__file__).parent / "system_prompt.txt"

if _prompt_file.exists():
    SYSTEM_PROMPT = _prompt_file.read_text(encoding="utf-8")
else:
    logger.warning("system_prompt.txt not found — using fallback prompt")
    SYSTEM_PROMPT = (
        "You are Ara, the AI assistant inside AskAra+. "
        "You help people across Southeast Asia understand government programs "
        "and services in simple language."
    )

# ---------------------------------------------------------------------------
# Tools that produce structured frontend data (step_cards / recommendations)
# ---------------------------------------------------------------------------

STRUCTURED_OUTPUT_TOOLS = {"summarize", "profile_match"}

# ---------------------------------------------------------------------------
# Greeting detection
# ---------------------------------------------------------------------------

GREETING_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|helo|hai|hola|salam|assalamualaikum|selamat|"
    r"terima\s*kasih|thank\s*you|thanks|ok|okay|got\s*it|noted|"
    r"good\s*morning|good\s*afternoon|good\s*evening|"
    r"sawadee|kumusta|magandang|apa\s*khabar|apa\s*kabar)\s*[.!?]*\s*$",
    re.IGNORECASE,
)


def _is_greeting(text: str) -> bool:
    return bool(GREETING_PATTERNS.match(text.strip()))


# ---------------------------------------------------------------------------
# Query classification — keyword heuristics (no LLM needed)
# ---------------------------------------------------------------------------

# TYPE D keywords — profiling / program discovery
PROFILING_PATTERNS = re.compile(
    r"("
    # Direct match for profiling structured format (from ProfilingFlow)
    r"country:\s*[A-Z]{2}.*situation:\s*\w+"
    # English phrases
    r"|find\s+(?:\w+\s+)?programs?\s+for\s+me"
    r"|what\s+(?:help|programs?|aid|benefits?)\s+can\s+I\s+get"
    r"|what\s+am\s+I\s+eligible\s+for"
    r"|am\s+I\s+eligible"
    # Malay
    r"|cari\s+program"
    r"|bantuan\s+apa\s+(?:yang\s+)?(?:saya|aku)"
    r"|kelayakan"
    r"|profil"
    # Program matching keywords
    r"|program\s+matching"
    # Identity + situation patterns
    r"|I\s+am\s+a\s+(?:worker|student|business|disaster)"
    r"|saya\s+(?:pekerja|pelajar|mangsa)"
    r")",
    re.IGNORECASE,
)

# TYPE C keywords — document scan (image/photo)
SCAN_PATTERNS = re.compile(
    r"(I\s+photographed|scanned?\s+document|saya\s+ambil\s+gambar|"
    r"document\s+type:|extracted\s+text:|Issuing\s+agency:)",
    re.IGNORECASE,
)

# TYPE B keywords — procedural (how-to)
# Broad patterns: any query about registering, applying, claiming, steps, etc.
PROCEDURAL_PATTERNS = re.compile(
    r"("
    # English
    r"how\s+(to|do\s+I|can\s+I|should\s+I)"
    r"|steps?\s+(to|for)"
    r"|process\s+(to|for|of)"
    r"|procedure"
    r"|apply\s+(for|to)"
    r"|register\s+(for|with|at)"
    r"|claim\s+(for|from|my)"
    r"|where\s+(to|do\s+I|can\s+I)\s+(go|apply|register|claim|get|submit)"
    r"|what\s+do\s+I\s+need\s+to"
    r"|what\s+documents?\s+(do\s+I\s+need|are\s+required|should\s+I)"
    # Malay — broad
    r"|cara\s+(nak|untuk|memohon|mendaftar|daftar|tuntut|buat|dapatkan|ambil)"
    r"|macam\s*mana\s*(nak|nok|untuk)?"
    r"|bagaimana"
    r"|langkah"
    r"|nak\s+(mohon|daftar|claim|tuntut|buat|dapatkan)"
    r"|nok\s+(daftar|mohon|buat|dapat)"
    r"|boleh\s+ke\s+(saya|aku)"
    r"|mohon|permohonan|mendaftar|pendaftaran"
    r"|tuntut(an)?|menuntut"
    # Indonesian
    r"|bagaimana\s+cara"
    r"|gimana\s+cara"
    r"|cara\s+daftar"
    r"|langkah.langkah"
    r"|persyaratan"
    # Filipino
    r"|paano"
    r"|mag-?apply|mag-?register|mag-?claim"
    r"|saan\s+(ako|po)\s+(mag|puwede)"
    r"|ano\s+ang\s+(kailangan|proseso|requirements?)"
    # Thai
    r"|วิธี|ขั้นตอน|สมัคร|ลงทะเบียน"
    r")",
    re.IGNORECASE,
)


def _classify_query(text: str) -> str:
    """Classify user query into TYPE A/B/C/D/E."""
    if _is_greeting(text):
        return "E"
    if SCAN_PATTERNS.search(text):
        return "C"
    if PROFILING_PATTERNS.search(text):
        return "D"
    if PROCEDURAL_PATTERNS.search(text):
        return "B"
    # Default: informational (TYPE A)
    return "A"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json_loads(data: str | dict | list) -> dict:
    """Universal sanitizer and normalizer for tool outputs."""
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


def _try_parse_structured(content) -> dict | None:
    """Try to parse structured output (step_cards / recommendations) from tool result."""
    if isinstance(content, str):
        data = _safe_json_loads(content)
    elif isinstance(content, dict):
        data = content
    else:
        data = _safe_json_loads(str(content))

    if isinstance(data, dict):
        if data.get("type") == "step_cards" and data.get("cards"):
            return data
        if data.get("type") == "recommendations" and data.get("items"):
            return data
    return None


# ---------------------------------------------------------------------------
# Search query cleaning — strip procedural noise, keep the subject
# ---------------------------------------------------------------------------

# Words/phrases that are NOT useful for semantic search
_NOISE_PATTERNS = re.compile(
    r"\b("
    r"how\s+(?:to|do\s+I|can\s+I|should\s+I)"
    r"|what\s+(?:is|are|do\s+I\s+need)"
    r"|where\s+(?:to|do\s+I|can\s+I)"
    r"|steps?\s+(?:to|for)"
    r"|process\s+(?:to|for|of)"
    r"|procedure\s+(?:to|for)"
    r"|cara\s+(?:nak|untuk|memohon|mendaftar|daftar|tuntut|buat|dapatkan)"
    r"|macam\s*mana\s*(?:nak|nok|untuk)?"
    r"|bagaimana\s*(?:cara)?"
    r"|gimana\s*(?:cara)?"
    r"|nak\s+(?:mohon|daftar|claim|tuntut|buat)"
    r"|nok\s+(?:daftar|mohon|buat|dapat)"
    r"|boleh\s+ke\s+(?:saya|aku)"
    r"|paano\s*(?:mag|po)?"
    r"|ano\s+ang\s+(?:kailangan|proseso)"
    r"|tolong|please|sila|help\s+me"
    r"|can\s+you|boleh\s+tak"
    r"|I\s+(?:want|need)\s+to"
    r"|saya\s+(?:nak|mau|ingin)"
    r")\b",
    re.IGNORECASE,
)

# Extra filler words to strip after noise removal
_FILLER_WORDS = re.compile(
    r"\b(the|a|an|for|to|in|at|on|my|me|I|di|ke|dari|yang|dan|atau|saya|aku|po|ko|ka|ba)\b",
    re.IGNORECASE,
)


def _clean_search_query(message: str) -> str:
    """Extract the core subject from a user message for better embedding search.

    e.g. "macam mana nak daftar SOCSO?" → "daftar SOCSO"
         "how to apply for flood relief?" → "apply flood relief"
    """
    cleaned = _NOISE_PATTERNS.sub("", message)
    # Remove punctuation
    cleaned = re.sub(r"[?!.,;:\"'()]+", " ", cleaned)
    # Strip filler words only if the result still has substance
    without_filler = _FILLER_WORDS.sub("", cleaned)
    without_filler = re.sub(r"\s+", " ", without_filler).strip()

    # If stripping filler words left almost nothing, keep the original cleaned version
    if len(without_filler) < 3:
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
    else:
        cleaned = without_filler

    # If still too short (e.g. "SOCSO"), that's fine — it's a focused query
    # If empty after all cleaning, fall back to original message
    return cleaned if cleaned else message.strip()


# ---------------------------------------------------------------------------
# Topic inference from keywords in the user message
# ---------------------------------------------------------------------------

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "worker_rights": [
        "socso", "perkeso", "employment", "worker", "pekerja", "majikan",
        "employer", "gaji", "salary", "wage", "kerja", "labour", "labor",
        "retrenchment", "dismissal", "buang kerja", "overtime", "kerja lebih",
        "kontrak", "contract", "bpjs ketenagakerjaan", "jamsostek",
        "owwa", "ofw", "dole", "migrant",
    ],
    "social_aid": [
        "bantuan", "aid", "welfare", "kebajikan", "jkm", "brim", "bsh",
        "sara", "bansos", "pkh", "sss", "pag-ibig", "pagibig",
        "subsidi", "subsidy", "b40", "miskin", "poor", "poverty",
        "4ps", "pantawid",
    ],
    "flood_relief": [
        "banjir", "flood", "bencana", "disaster", "mangsa", "victim",
        "pemindahan", "evacuation", "relief", "wang ihsan",
        "nadma", "fema",
    ],
    "health": [
        "kesihatan", "health", "hospital", "clinic", "klinik", "doctor",
        "doktor", "mysejahtera", "bpjs kesehatan", "philhealth",
        "vaksin", "vaccine", "denggi", "dengue", "insurance",
        "perlindungan", "coverage", "medical",
    ],
    "business_support": [
        "business", "perniagaan", "usahawan", "entrepreneur", "sme",
        "grant", "geran", "loan", "pinjaman", "dti", "tekun", "mara",
        "lesen", "license", "permit", "ssm", "registration",
    ],
    "education": [
        "education", "pendidikan", "scholarship", "biasiswa", "sekolah",
        "school", "university", "universiti", "ptptn", "training",
        "latihan", "tesda", "skills",
    ],
}


def _infer_topic(text: str) -> str:
    """Infer the most likely ChromaDB topic from keywords in the message.
    Returns empty string if no confident match."""
    text_lower = text.lower()
    scores: dict[str, int] = {}

    for topic, keywords in _TOPIC_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            scores[topic] = count

    if not scores:
        return ""

    # Return topic with highest keyword count (ties go to first alphabetically)
    best_topic = max(scores, key=scores.get)  # type: ignore
    logger.debug("Topic inferred: %s (scores=%s)", best_topic, scores)
    return best_topic


# ---------------------------------------------------------------------------
# Relevance check — are search results about what the user asked?
# ---------------------------------------------------------------------------

def _check_relevance(search_result: dict, user_query: str, inferred_topic: str) -> bool:
    """Quick heuristic: do the top results seem relevant to the query subject?

    Returns True if results look relevant, False if they seem off-topic.
    Checks:
    1. If user mentions a specific program/entity, it must appear in results.
    2. If topic was inferred, at least one top result should match that topic.
    3. Fallback: keyword overlap between query and top result text.
    """
    results = search_result.get("results", [])
    if not results:
        return False

    # ── Check 1: Specific entity / program name match ──
    # If the user mentions a known program, the results MUST mention it too
    # (using any of its aliases). Otherwise we're returning the wrong document.
    entity = _extract_entity(user_query)
    if entity:
        aliases = _KNOWN_ENTITIES.get(entity, [entity])
        # Check if any top-3 result mentions any alias of this entity
        entity_found = False
        for r in results[:3]:
            text = r.get("text", "").lower()
            title = r.get("source", {}).get("document_title", "").lower()
            combined = text + " " + title
            if any(alias in combined for alias in aliases):
                entity_found = True
                break
        if not entity_found:
            logger.info(
                "Relevance check FAILED: entity '%s' (aliases=%s) not found in top results",
                entity, aliases,
            )
            return False

    # ── Check 2: Topic match ──
    if inferred_topic:
        topic_matched = any(
            r.get("source", {}).get("topic", "") == inferred_topic
            for r in results[:3]
        )
        if topic_matched:
            return True

    # If we couldn't infer a topic, trust the similarity score
    if not inferred_topic:
        return True

    # ── Check 3: Keyword overlap fallback ──
    top_text = results[0].get("text", "").lower()
    query_words = [w for w in user_query.lower().split() if len(w) > 3]
    overlap = sum(1 for w in query_words if w in top_text)

    if overlap >= 2 or (len(query_words) <= 2 and overlap >= 1):
        return True

    logger.info(
        "Relevance check FAILED: inferred_topic=%s, top_doc_topic=%s, query_word_overlap=%d",
        inferred_topic,
        results[0].get("source", {}).get("topic", ""),
        overlap,
    )
    return False


# ---------------------------------------------------------------------------
# Entity / program name extraction
# ---------------------------------------------------------------------------

# Known ASEAN government programs and agencies — if a user mentions these,
# the search results MUST contain them or we fall back to gov portal.
_KNOWN_ENTITIES: dict[str, list[str]] = {
    # Malaysia
    "socso": ["socso", "perkeso", "keselamatan sosial"],
    "perkeso": ["socso", "perkeso", "keselamatan sosial"],
    "epf": ["epf", "kwsp", "kumpulan wang simpanan"],
    "kwsp": ["epf", "kwsp", "kumpulan wang simpanan"],
    "jkm": ["jkm", "jabatan kebajikan masyarakat", "kebajikan"],
    "brim": ["brim", "bsh", "bantuan sara hidup", "sumbangan tunai"],
    "mysejahtera": ["mysejahtera"],
    "mara": ["mara", "majlis amanah rakyat"],
    "ptptn": ["ptptn"],
    # Indonesia
    "bpjs": ["bpjs", "jaminan sosial"],
    "ktp": ["ktp", "kartu tanda penduduk"],
    "bansos": ["bansos", "bantuan sosial"],
    "pkh": ["pkh", "program keluarga harapan"],
    # Philippines
    "philhealth": ["philhealth"],
    "sss": ["sss", "social security system"],
    "pagibig": ["pag-ibig", "pagibig", "hdmf"],
    "owwa": ["owwa"],
    "dti": ["dti", "department of trade"],
    "tesda": ["tesda"],
    "4ps": ["4ps", "pantawid"],
    # Thailand
    "ประกันสังคม": ["ประกันสังคม", "social security"],
    "บัตรทอง": ["บัตรทอง", "30 baht"],
}


def _extract_entity(query: str) -> str | None:
    """Check if the user's query mentions a known government program or agency.
    Returns the canonical lowercase name, or None if no match."""
    query_lower = query.lower()
    for entity_key in _KNOWN_ENTITIES:
        if entity_key in query_lower:
            return entity_key
    return None


# ---------------------------------------------------------------------------
# URL hallucination guard
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(r'https?://[^\s)\]>"\']+')


def _strip_hallucinated_urls(text: str, allowed_urls: set[str]) -> str:
    """Remove any URL from text that isn't in the allowed set."""
    if not allowed_urls:
        return URL_PATTERN.sub("[link removed]", text)

    def replace_url(match: re.Match) -> str:
        url = match.group(0).rstrip(".,;:!?)")
        for allowed in allowed_urls:
            if url.startswith(allowed) or allowed.startswith(url):
                return match.group(0)
        return "[link removed]"

    return URL_PATTERN.sub(replace_url, text)


# ---------------------------------------------------------------------------
# Source extraction helpers
# ---------------------------------------------------------------------------

def _extract_sources_from_search(search_result: dict) -> list[dict]:
    """Extract source citations from search_documents result."""
    sources = []
    for r in search_result.get("results", []):
        src = r.get("source", {})
        if src:
            entry = {
                "title": src.get("document_title", ""),
                "url": src.get("document_url", ""),
                "source_agency": src.get("source_agency", ""),
                "country": src.get("country", ""),
                "relevance": r.get("similarity", 0),
            }
            if entry["title"] and entry not in sources:
                sources.append(entry)
    return sources


def _extract_sources_from_portal(portal_result: dict) -> list[dict]:
    """Extract source citations from fetch_gov_portal result."""
    sources = []
    for r in portal_result.get("results", []):
        entry = {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "source_agency": "",
            "country": portal_result.get("country", ""),
        }
        if entry["title"] and entry not in sources:
            sources.append(entry)
    return sources


def _collect_allowed_urls(sources: list[dict]) -> set[str]:
    """Collect all URLs from sources for hallucination guard."""
    return {s.get("url", "") for s in sources if s.get("url")}


def _build_context_hint(country: str | None, language: str | None) -> str:
    """Build context hint to append to user message."""
    parts = []
    if country:
        parts.append(f"country={country}")
    if language:
        parts.append(f"language={language}")
    if parts:
        return f"\n\n[Context: {' '.join(parts)}]"
    return ""


def _get_text_from_search(result: dict) -> str:
    """Combine document texts from search results."""
    texts = []
    for r in result.get("results", []):
        text = r.get("text", "")
        if text:
            texts.append(text)
    return "\n\n".join(texts)


def _has_good_results(result: dict) -> bool:
    """Check if search results are relevant (not empty/low-confidence)."""
    status = result.get("status", "")
    if status in ("no_results", "low_confidence", "error"):
        return False
    results = result.get("results", [])
    return len(results) > 0


def _extract_profiling_data(text: str) -> dict | None:
    """Try to extract profiling data (country/situation/need) from a structured message.

    Handles both:
    - Structured: "country: MY, situation: disaster_victim, need: financial_aid"
    - Natural:    "I am in Malaysia (MY). My situation: disaster affected. I need: financial aid."
    """
    # ── Country — match ISO code ──
    country_match = re.search(r'country[:\s]+([A-Z]{2})', text, re.IGNORECASE)
    if not country_match:
        # Fallback: "I am in Malaysia (MY)"
        country_match = re.search(r'\(([A-Z]{2})\)', text)
    if not country_match:
        return None

    # ── Situation — exact enum values first, then fuzzy mapping ──
    situation_match = re.search(
        r'situation[:\s]+(worker|business_owner|family|disaster_victim|unemployed|student)',
        text, re.IGNORECASE,
    )
    if not situation_match:
        # Fuzzy fallback: map human labels to enum values
        situation_map = {
            "worker": "worker",
            "business owner": "business_owner",
            "business": "business_owner",
            "family": "family",
            "resident": "family",
            "disaster": "disaster_victim",
            "disaster affected": "disaster_victim",
            "flood": "disaster_victim",
            "unemployed": "unemployed",
            "student": "student",
        }
        text_lower = text.lower()
        found_situation = None
        for key, val in situation_map.items():
            if key in text_lower:
                found_situation = val
                break
        if not found_situation:
            return None
    else:
        found_situation = situation_match.group(1).lower()

    # ── Need — exact enum values first, then fuzzy mapping ──
    need_match = re.search(
        r'need[:\s]+(financial_aid|healthcare|worker_rights|business_support|housing|legal_aid|education)',
        text, re.IGNORECASE,
    )
    if not need_match:
        # Fuzzy fallback: map human labels to enum values
        need_map = {
            "financial aid": "financial_aid",
            "financial": "financial_aid",
            "healthcare": "healthcare",
            "health": "healthcare",
            "medical": "healthcare",
            "worker rights": "worker_rights",
            "employment": "worker_rights",
            "legal rights": "legal_aid",
            "legal aid": "legal_aid",
            "legal": "legal_aid",
            "business support": "business_support",
            "housing": "housing",
            "education": "education",
        }
        text_lower = text.lower()
        found_need = None
        for key, val in need_map.items():
            if key in text_lower:
                found_need = val
                break
        if not found_need:
            return None
    else:
        found_need = need_match.group(1).lower()

    return {
        "country": country_match.group(1).upper(),
        "situation": found_situation,
        "need": found_need,
    }


# ---------------------------------------------------------------------------
# Core: Streaming deterministic pipeline
# ---------------------------------------------------------------------------

async def run_agent_streaming(
    message: str,
    *,
    country: str | None = None,
    language: str | None = None,
    history: list[dict] | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Deterministic tool pipeline with streaming response.

    Yields dicts:
        {"type": "token",      "content": "..."}
        {"type": "tool_start", "content": "..."}
        {"type": "tool_end",   "content": "..."}
        {"type": "structured", "content": {...}}
        {"type": "sources",    "content": [...]}
        {"type": "done",       "content": "..."}
    """
    tool_calls: list[str] = []
    sources: list[dict] = []
    structured = None
    full_response = ""

    try:
        # ── TYPE E: Greeting shortcut ──────────────────────────────
        query_type = _classify_query(message)
        logger.info("Query classified as TYPE %s: '%s'", query_type, message[:80])

        if query_type == "E":
            async for token in call_llm_streaming(
                message,
                system_prompt=(
                    "You are Ara, a warm multilingual assistant for ASEAN government services. "
                    "Respond to the greeting briefly and warmly. Offer to help with government "
                    "services. Match the user's language. Keep it to 1-2 sentences."
                ),
                history=history,
            ):
                full_response += token
                yield {"type": "token", "content": token}
            yield {"type": "done", "content": full_response}
            return

        # ── TYPE D: Program Matching ───────────────────────────────
        if query_type == "D":
            profiling_data = _extract_profiling_data(message)

            if profiling_data:
                yield {"type": "tool_start", "content": "profile_match"}
                tool_calls.append("profile_match")

                result_json = await _profile_match(
                    country=profiling_data["country"],
                    situation=profiling_data["situation"],
                    need=profiling_data["need"],
                )
                result = _safe_json_loads(result_json)
                yield {"type": "tool_end", "content": "profile_match"}

                s = _try_parse_structured(result)
                if s:
                    structured = s
                    yield {"type": "structured", "content": structured}

                # Short intro via LLM
                profile_summary = result.get("profile_summary", "your profile")
                llm_prompt = (
                    f"The user asked for program recommendations. Based on {profile_summary}, "
                    f"I found {result.get('total_matches', 0)} matching programs. "
                    f"Write a warm 1-2 sentence intro in the user's language. "
                    f"Do NOT list the programs — the cards display them automatically.\n\n"
                    f"User message: {message}"
                )

                async for token in call_llm_streaming(
                    llm_prompt,
                    system_prompt=SYSTEM_PROMPT,
                    history=history,
                ):
                    full_response += token
                    yield {"type": "token", "content": token}

                yield {"type": "done", "content": full_response}
                return
            # If no structured profiling data found, fall through to TYPE A

        # ── TYPE C: Document Scan ──────────────────────────────────
        if query_type == "C":
            # Text already extracted by /api/scan endpoint
            # The message contains "Extracted text: ..." from CameraCapture
            extracted_text = message

            # Step 1: Detect language on the extracted text
            yield {"type": "tool_start", "content": "detect_language"}
            tool_calls.append("detect_language")
            lang_json = _detect_language(text=extracted_text)
            lang_data = json.loads(lang_json)
            yield {"type": "tool_end", "content": "detect_language"}

            detected_lang = lang_data.get("primary_lang", "en")
            detected_country = lang_data.get("country_hint", "") or (country or "")

            # Step 2: Simplify the extracted text
            yield {"type": "tool_start", "content": "simplify"}
            tool_calls.append("simplify")
            simplified_json = await _simplify_text(
                text=extracted_text,
                target_grade_level=5,
                country=detected_country,
                language=detected_lang,
            )
            simplified_data = _safe_json_loads(simplified_json)
            simplified_text = simplified_data.get("simplified_text", extracted_text)
            yield {"type": "tool_end", "content": "simplify"}

            # Step 3: Generate explanation via LLM
            llm_prompt = (
                f"A user photographed a government document. I extracted and simplified the text.\n\n"
                f"Simplified text:\n{simplified_text}\n\n"
                f"Explain what this document says in simple terms. "
                f"Tell the user what it means for them and what they should do next. "
                f"Use {detected_lang} language. Keep it simple and warm.\n\n"
                f"User's original message: {message}"
            )

            async for token in call_llm_streaming(
                llm_prompt,
                system_prompt=SYSTEM_PROMPT,
                history=history,
            ):
                full_response += token
                yield {"type": "token", "content": token}

            yield {"type": "done", "content": full_response}
            return

        # ── TYPE A (Informational) & TYPE B (Procedural) ──────────
        # Both share the same initial pipeline: detect → search → simplify
        # TYPE B adds: summarize into step_cards

        # Step 1: Detect language
        yield {"type": "tool_start", "content": "detect_language"}
        tool_calls.append("detect_language")
        lang_json = _detect_language(text=message)
        lang_data = json.loads(lang_json)
        yield {"type": "tool_end", "content": "detect_language"}

        detected_lang = lang_data.get("primary_lang", "en")
        detected_dialect = lang_data.get("dialect", "standard")
        detected_country = lang_data.get("country_hint", "") or (country or "")
        is_code_mixed = lang_data.get("is_code_mixed", False)

        logger.info(
            "Language detected: lang=%s, dialect=%s, country=%s, code_mixed=%s",
            detected_lang, detected_dialect, detected_country, is_code_mixed,
        )

        # Step 2: Search knowledge base
        # Clean the query for better embedding match and infer topic
        search_query = _clean_search_query(message)
        inferred_topic = _infer_topic(message)

        logger.info(
            "Search: raw='%s' → cleaned='%s', inferred_topic='%s'",
            message[:60], search_query[:60], inferred_topic,
        )

        yield {"type": "tool_start", "content": "search_documents"}
        tool_calls.append("search_documents")
        search_json = _search_documents(
            query=search_query,
            country=detected_country,
            topic=inferred_topic,
        )
        search_result = _safe_json_loads(search_json)
        yield {"type": "tool_end", "content": "search_documents"}

        # If topic-filtered search got nothing, retry without topic filter
        if not _has_good_results(search_result) and inferred_topic:
            logger.info("No results with topic=%s — retrying without topic filter", inferred_topic)
            search_json = _search_documents(
                query=search_query,
                country=detected_country,
                topic="",
            )
            search_result = _safe_json_loads(search_json)

        sources.extend(_extract_sources_from_search(search_result))

        # Step 2b: Relevance check + fallback
        info_text = ""
        source_tier = "knowledge_base"

        has_results = _has_good_results(search_result)
        is_relevant = has_results and _check_relevance(search_result, search_query, inferred_topic)

        if has_results and is_relevant:
            info_text = _get_text_from_search(search_result)
        else:
            if has_results and not is_relevant:
                logger.info("Search results failed relevance check — trying gov portal")
            else:
                logger.info("Knowledge base had no good results — trying gov portal")
            yield {"type": "tool_start", "content": "fetch_gov_portal"}
            tool_calls.append("fetch_gov_portal")
            portal_json = await _fetch_gov_portal(url=message, country=detected_country)
            portal_result = _safe_json_loads(portal_json)
            yield {"type": "tool_end", "content": "fetch_gov_portal"}

            sources.extend(_extract_sources_from_portal(portal_result))
            info_text = portal_result.get("content", "")
            source_tier = portal_result.get("source_tier", "web")

        # Step 3: Simplify the retrieved text
        if info_text:
            yield {"type": "tool_start", "content": "simplify"}
            tool_calls.append("simplify")
            simplified_json = await _simplify_text(
                text=info_text[:3000],  # Cap input length
                target_grade_level=5,
                country=detected_country,
                language=detected_lang,
            )
            simplified_data = _safe_json_loads(simplified_json)
            simplified_text = simplified_data.get("simplified_text", info_text)
            yield {"type": "tool_end", "content": "simplify"}
        else:
            simplified_text = ""

        # Step 4 (TYPE B only): Summarize into step cards
        if query_type == "B" and simplified_text:
            yield {"type": "tool_start", "content": "summarize"}
            tool_calls.append("summarize")
            summary_json = await _summarize_text(
                text=simplified_text,
                format="step_cards",
                language=detected_lang,
                max_steps=5,
            )
            yield {"type": "tool_end", "content": "summarize"}

            s = _try_parse_structured(summary_json)
            if s:
                structured = s
                yield {"type": "structured", "content": structured}
                logger.info("Step cards generated: %d cards", len(s.get("cards", [])))

        # Step 5 (optional): Translate if user language != document language
        if detected_lang not in ("en", "") and source_tier == "knowledge_base":
            # Check if we need translation
            doc_lang = search_result.get("results", [{}])[0].get("source", {}).get("language", "")
            if doc_lang and doc_lang != detected_lang:
                yield {"type": "tool_start", "content": "translate"}
                tool_calls.append("translate")
                translate_json = await _translate_text(
                    text=simplified_text[:2000],
                    source_lang=doc_lang,
                    target_lang=detected_lang,
                )
                translate_data = _safe_json_loads(translate_json)
                translated = translate_data.get("translated_text", "")
                if translated:
                    simplified_text = translated
                yield {"type": "tool_end", "content": "translate"}

        # Step 6 (optional): Dialect adaptation
        if detected_dialect != "standard":
            dialect_key_map = {
                "kelantan": "kelantan_malay",
                "javanese": "javanese",
                "waray": "waray",
                "kham_mueang": "kham_mueang",
            }
            dialect_key = dialect_key_map.get(detected_dialect)
            if dialect_key:
                yield {"type": "tool_start", "content": "dialect_adapt"}
                tool_calls.append("dialect_adapt")
                dialect_json = await _dialect_adapt(
                    text=simplified_text[:2000],
                    target_dialect=dialect_key,
                )
                dialect_data = _safe_json_loads(dialect_json)
                adapted = dialect_data.get("adapted_text", "")
                if adapted:
                    simplified_text = adapted
                yield {"type": "tool_end", "content": "dialect_adapt"}

        # ── Emit sources before LLM response ──
        if sources:
            yield {"type": "sources", "content": sources}

        # ── Step FINAL: Generate response via LLM ──────────────────
        if query_type == "B" and structured:
            # TYPE B with step cards: only a short intro
            llm_prompt = (
                f"I found information and created step-by-step cards for the user.\n\n"
                f"Simplified info:\n{simplified_text[:1000]}\n\n"
                f"Write ONLY a warm 1-2 sentence intro acknowledging their question. "
                f"Do NOT list the steps — the step cards display automatically. "
                f"Use {detected_lang} language. Match the user's language.\n\n"
                f"User question: {message}"
            )
        elif simplified_text:
            source_disclaimer = ""
            if source_tier == "web":
                source_disclaimer = (
                    "\n\nIMPORTANT: Add a note that this info is from web search "
                    "and should be verified with the relevant agency."
                )
            llm_prompt = (
                f"Here is the simplified government information to answer the user's question:\n\n"
                f"{simplified_text[:2000]}\n\n"
                f"Write a clear, warm response using this information. "
                f"Use simple language (Grade 5 level). "
                f"Use {detected_lang} language. Match the user's language."
                f"{source_disclaimer}\n\n"
                f"User question: {message}"
            )
        else:
            # No info found
            llm_prompt = (
                f"I searched the knowledge base and government portals but could not find "
                f"specific information for this question. "
                f"Write a helpful response acknowledging you couldn't find exact info. "
                f"Suggest the user contact the relevant government agency directly. "
                f"Use {detected_lang} language. Match the user's language.\n\n"
                f"User question: {message}"
            )

        async for token in call_llm_streaming(
            llm_prompt,
            system_prompt=SYSTEM_PROMPT,
            history=history,
        ):
            full_response += token
            yield {"type": "token", "content": token}

        # Post-process: strip hallucinated URLs
        allowed_urls = _collect_allowed_urls(sources)
        full_response = _strip_hallucinated_urls(full_response, allowed_urls)

        yield {"type": "done", "content": full_response}

    except Exception as exc:
        logger.error("Agent pipeline error: %s", exc, exc_info=True)
        yield {
            "type": "error",
            "content": "I'm sorry, I encountered an issue. Please try again.",
        }


# ---------------------------------------------------------------------------
# Core: Non-streaming agent run
# ---------------------------------------------------------------------------

async def run_agent(
    message: str,
    *,
    country: str | None = None,
    language: str | None = None,
    history: list[dict] | None = None,
) -> dict:
    """
    Non-streaming version. Collects all events from the streaming pipeline.

    Returns dict with: reply, sources, tool_calls, structured
    """
    tool_calls: list[str] = []
    sources: list[dict] = []
    structured = None
    full_response = ""

    async for event in run_agent_streaming(
        message,
        country=country,
        language=language,
        history=history,
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


# ---------------------------------------------------------------------------
# Cleanup — no MCP client to close, but keep the interface
# ---------------------------------------------------------------------------

async def cleanup_agent() -> None:
    """Cleanup hook (called on app shutdown). No-op in deterministic pipeline."""
    logger.info("Agent cleanup complete (deterministic pipeline — nothing to close).")