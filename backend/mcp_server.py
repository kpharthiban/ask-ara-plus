"""
AskAra+ MCP Server
====================
FastMCP server exposing 11 tools to the LangChain ReAct agent.
The agent dynamically selects which tools to call based on user queries.

Architecture:
  - This file: MCP tool registration + thin wrappers (type hints & docstrings)
  - tools/*.py: Actual implementation logic (stubs now, real later)
  - db.py: ChromaDB singleton (already set up)
  - llm_client.py: LLM API wrapper (already set up)

Running standalone (for testing):
  uv run python mcp_server.py

Day 2 morning (with Lineysha):
  - Wire this to LangChain ReAct agent via langchain-mcp-adapters
  - Test: agent receives message → calls search_documents → gets result
  - Fallback: wrap tools as regular LangChain Tool objects if adapter is buggy

Tool ownership:
  Pharthiban:  search_documents (Day 2), detect_language (Day 2),
               scan_document (Day 5), fetch_gov_portal (Tier 3)
  Lineysha:    simplify (Day 3), summarize (Day 3), assess_complexity (Day 3),
               translate (Day 4), dialect_adapt (Day 4), profile_match (Day 5)
  TBD:         text_to_speech (may stay on frontend)
"""

from fastmcp import FastMCP
import logging

# ── Import tool implementations from tools/ ───────────────────
from tools.search import search_documents as _search_documents
from tools.language import detect_language as _detect_language
from tools.simplify import simplify_text as _simplify_text
from tools.translate import translate_text as _translate_text
from tools.summarize import summarize_text as _summarize_text
from tools.dialect import dialect_adapt as _dialect_adapt
from tools.complexity import assess_complexity as _assess_complexity
from tools.speech import text_to_speech as _text_to_speech
from tools.scanner import scan_document as _scan_document
from tools.portal import fetch_gov_portal as _fetch_gov_portal
from tools.profiler import profile_match as _profile_match

logging.basicConfig(level=logging.INFO) # Configures the root logger
logger = logging.getLogger(__name__)

# ── Initialize FastMCP server ─────────────────────────────────
mcp = FastMCP(
    "AskAra+",
    instructions=(
        "AskAra+ is a multilingual ASEAN government services assistant. "
        "Use these tools to help citizens understand government benefits, "
        "worker rights, health services, and disaster relief across Malaysia, "
        "Indonesia, Philippines, and Thailand. Always detect the user's "
        "language first, then search, simplify, and translate as needed."
    ),
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TIER 1 — MUST SHIP (Core tools for basic RAG pipeline)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@mcp.tool()
def search_documents(query: str, country: str = "") -> str:
    """Search the AskAra+ knowledge base for relevant government documents.

    Use this when a user asks about government services, policies, benefits,
    worker rights, health services, or disaster relief. Supports queries in
    any ASEAN language — cross-lingual retrieval is built in.

    Args:
        query: The user's question or search terms (any language).
        country: Filter by country — "MY", "ID", "PH", "TH", or "ASEAN". Empty = all.

    Returns:
        JSON with ranked document chunks, source citations, and similarity scores.
    """
    logger.info(f"search_documents called: query='{query}', country='{country}'")
    return _search_documents(query=query, country=country, topic="")


@mcp.tool()
def detect_language(text: str) -> str:
    """Detect the primary language, dialect, and code-mixing in user input.

    ALWAYS call this FIRST on every new user message. The result determines:
    - Which language to search/respond in
    - Whether translation is needed
    - Whether dialect adaptation should be applied
    - Which LLM model to route to

    Args:
        text: Raw user input text.

    Returns:
        JSON with primary_lang, dialect, secondary_langs, is_code_mixed,
        confidence, and recommended_llm.
    """
    return _detect_language(text=text)


@mcp.tool()
async def simplify(text: str, target_grade_level: int = 5, country: str = "", language: str = "") -> str:
    """Simplify complex official text to a target reading grade level.

    Use after retrieving documents. Replaces jargon using the country-specific
    glossary, then rewrites with an LLM for short, clear sentences.

    Args:
        text: The complex official text to simplify.
        target_grade_level: Target reading level (1-12). Default 5 = primary school.
        country: Country code for glossary selection ("MY", "ID", "PH", "TH").
        language: Language of the text ("ms", "id", "tl", "th", "en").

    Returns:
        JSON with simplified_text, terms_replaced, and grade levels.
    """
    return await _simplify_text(
        text=text,
        target_grade_level=target_grade_level,
        country=country,
        language=language,
    )


@mcp.tool()
async def translate(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text between SEA languages.

    Use when the user's language differs from the document language.
    Example: Indonesian worker asks about Malaysian SOCSO → retrieve in
    Malay → simplify → translate to Indonesian.

    Supported: ms, en, id, tl, th, jv, ceb, war.

    Args:
        text: Text to translate (ideally already simplified).
        source_lang: Source language code.
        target_lang: Target language code.

    Returns:
        JSON with translated_text, language codes, and model_used.
    """
    return await _translate_text(text=text, source_lang=source_lang, target_lang=target_lang)


@mcp.tool()
async def summarize(text: str, format: str = "step_cards", language: str = "en", max_steps: int = 5) -> str:
    """Summarize text into actionable Step Cards or bullet points.

    Use after simplification to convert procedures into swipeable Step Cards
    (the primary output for how-to guides) or concise bullets (for facts).

    Step Cards include: step number, title, body, and an action button
    (link, phone number, or address).

    Args:
        text: Text to summarize (ideally already simplified).
        format: "step_cards" (default, for procedures) or "bullets" (for info).
        language: Output language code.

    Returns:
        JSON — step_cards: {type, cards: [{step, total, title, body, action}]}
        or bullets: {type, points: ["...", ...]}.
    """
    return await _summarize_text(text=text, format=format, language=language, max_steps=max_steps)


@mcp.tool()
def assess_complexity(text: str) -> str:
    """Assess the reading complexity of text.

    Call this before simplify() to decide if simplification is needed.
    If grade_level > 6, the text should be simplified before showing to user.

    Args:
        text: Text to assess.

    Returns:
        JSON with grade_level, difficult_terms, jargon_count, and suggestion.
    """
    return _assess_complexity(text=text)


@mcp.tool()
async def dialect_adapt(text: str, target_dialect: str) -> str:
    """Adapt standard language text to a regional dialect.

    Use as the FINAL step before presenting text, only when the user's
    detected dialect is non-standard (e.g., Kelantan Malay, Javanese).

    Supported: kelantan_malay, javanese, cebuano, waray.

    Args:
        text: Simplified text in the base standard language.
        target_dialect: Target dialect identifier.

    Returns:
        JSON with adapted_text, dialect info, and words_adapted count.
    """
    return await _dialect_adapt(text=text, target_dialect=target_dialect)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TIER 2 — STRONG DIFFERENTIATORS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@mcp.tool()
async def scan_document(image_base64: str, source_hint: str = "") -> str:
    """OCR a photographed government document (Snap & Understand).

    User photographs a letter, form, or notice → this extracts text,
    identifies the document type and issuing agency, and pulls out
    key info (dates, amounts, reference numbers).

    Args:
        image_base64: Base64-encoded image from camera (JPEG/PNG).
        source_hint: Optional hint like "PERKESO letter" for better accuracy.

    Returns:
        JSON with extracted_text, document_type, detected_language,
        issuing_agency, key_info, and confidence score.
    """
    return await _scan_document(image_base64=image_base64, source_hint=source_hint)


@mcp.tool()
async def profile_match(country: str, situation: str, need: str) -> str:
    """Match user profile to eligible government programs (Proactive Agent).

    Powers Mode 2 of the agent. Instead of waiting for the user to ask
    the right question, proactively recommends programs they qualify for.

    Args:
        country: "MY", "ID", "PH", "TH".
        situation: "worker", "business_owner", "family", "disaster_victim",
                   "unemployed", "student".
        need: "financial_aid", "healthcare", "worker_rights",
              "business_support", "housing", "legal_aid", "education".

    Returns:
        JSON with type="recommendations", matches array (program_name,
        description, who_qualifies, relevance_score), and profile_summary.
    """
    return await _profile_match(country=country, situation=situation, need=need)


@mcp.tool()
def text_to_speech(text: str, language: str = "en") -> str:
    """Convert text to speech audio (server-side fallback).

    NOTE: Frontend already handles TTS via browser Web Speech API.
    Only call this if browser TTS is unavailable or quality is poor.

    Args:
        text: Text to speak (keep under 500 chars).
        language: Voice language code.

    Returns:
        JSON with audio_url, duration, and engine info.
    """
    return _text_to_speech(text=text, language=language)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TIER 3 — NICE TO HAVE (cut if behind schedule)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@mcp.tool()
async def fetch_gov_portal(query: str, country: str = "") -> str:
    """Search government websites for fresh information using DuckDuckGo.

    Use this when search_documents returned no results or irrelevant results.
    Pass a descriptive search query (NOT a URL).

    Args:
        query: Descriptive search query about the government service or program.
               Example: "bantuan banjir pendaftaran Malaysia"
               Example: "SOCSO claim process worker benefits"
        country: Country code — "MY", "ID", "PH", "TH". Scopes search to gov domains.

    Returns:
        JSON with results array (title, url, snippet), combined content, and source_tier.
    """
    return await _fetch_gov_portal(url=query, country=country)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Server entry point (for standalone testing)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    REGISTERED_TOOLS = [
        "search_documents",
        "detect_language",
        "simplify",
        "translate",
        "summarize",
        "assess_complexity",
        "dialect_adapt",
        "text_to_speech",
        "scan_document",
        "fetch_gov_portal",
        "profile_match",
    ]

    print("Starting AskAra+ MCP Server...")
    print(f"Registered {len(REGISTERED_TOOLS)} tools:")
    for name in REGISTERED_TOOLS:
        print(f"  - {name}")
    print("\nRunning on http://0.0.0.0:8001 ...")

    mcp.run(transport="streamable-http", host="0.0.0.0", port=8001)