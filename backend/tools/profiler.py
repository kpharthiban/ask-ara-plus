"""
profile_match — Match user profile to eligible government programs.

Owner: Pharthiban (ChromaDB wiring), Lineysha (ranking logic/prompts)
Depends on: db.py (ChromaDB), llm_client.py

This powers the Proactive Eligibility Agent (Mode 2):
1. Agent greets user and asks 3 profiling questions
2. User provides: country, situation, need
3. This tool cross-references against ALL documents in the knowledge base
4. Returns ranked list of matching programs as recommendation cards

Note: This function is async because it calls async call_llm.
"""

import json
import logging
from pathlib import Path
import sys

# Ensure backend/ is importable for db module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("askara.profiler")


# ── Profile categories ────────────────────────────────────────
SITUATIONS = [
    "worker",           # Employed / migrant worker
    "business_owner",   # Small business / entrepreneur
    "family",           # Family / dependents
    "disaster_victim",  # Natural disaster affected
    "unemployed",       # Job seeker
    "student",          # Student / trainee
]

NEEDS = [
    "financial_aid",    # Cash assistance, subsidies
    "healthcare",       # Health insurance, medical help
    "worker_rights",    # Employment law, compensation
    "business_support", # Grants, loans, training
    "housing",          # Shelter, housing assistance
    "legal_aid",        # Legal representation, rights info
    "education",        # Training, scholarships
]

# ── Situation+Need → topic mapping ────────────────────────────
# Maps (situation, need) pairs to ChromaDB topic filters.
# Falls back to the need alone if no specific mapping exists.
TOPIC_MAP: dict[tuple[str, str], str] = {
    ("worker", "financial_aid"): "worker_rights",
    ("worker", "healthcare"): "health",
    ("worker", "worker_rights"): "worker_rights",
    ("worker", "legal_aid"): "worker_rights",
    ("business_owner", "financial_aid"): "business_support",
    ("business_owner", "business_support"): "business_support",
    ("family", "financial_aid"): "social_aid",
    ("family", "healthcare"): "health",
    ("family", "housing"): "social_aid",
    ("disaster_victim", "financial_aid"): "flood_relief",
    ("disaster_victim", "housing"): "flood_relief",
    ("disaster_victim", "healthcare"): "flood_relief",
    ("unemployed", "financial_aid"): "social_aid",
    ("unemployed", "education"): "education",
    ("student", "financial_aid"): "education",
    ("student", "education"): "education",
}

# ── Search query templates by situation ───────────────────────
QUERY_TEMPLATES: dict[str, str] = {
    "worker": "{need_phrase} for workers employees",
    "business_owner": "{need_phrase} for small business entrepreneurs",
    "family": "{need_phrase} for families dependents",
    "disaster_victim": "{need_phrase} disaster relief flood aid",
    "unemployed": "{need_phrase} unemployment job seeker benefits",
    "student": "{need_phrase} student scholarship training",
}

NEED_PHRASES: dict[str, str] = {
    "financial_aid": "financial assistance cash aid subsidy",
    "healthcare": "health insurance medical coverage",
    "worker_rights": "worker rights employment protection compensation",
    "business_support": "business grant loan support program",
    "housing": "housing shelter accommodation assistance",
    "legal_aid": "legal aid rights protection",
    "education": "education training scholarship program",
}

COUNTRY_NAMES: dict[str, str] = {
    "MY": "Malaysia",
    "ID": "Indonesia",
    "PH": "Philippines",
    "TH": "Thailand",
}

# Icons for recommendation cards (by need category)
NEED_ICONS: dict[str, str] = {
    "financial_aid": "💰",
    "healthcare": "🏥",
    "worker_rights": "⚖️",
    "business_support": "🏢",
    "housing": "🏠",
    "legal_aid": "📜",
    "education": "🎓",
}

# ── LLM ranking prompt ───────────────────────────────────────
RANKING_SYSTEM_PROMPT = """\
You are AskAra+, a government services advisor for Southeast Asia.

Given a user profile and a set of document chunks from the knowledge base,
identify the most relevant government programs and rank them by how well
they match the user's situation and needs.

Rules:
1. Only include programs that genuinely match the user's profile.
2. For each program, give a clear 1-sentence description in simple language.
3. Explain who qualifies in simple terms.
4. Assign a relevance score from 0.0 to 1.0.
5. Return EXACTLY valid JSON — no markdown, no explanations.
6. Return at most 4 programs. If fewer match, return fewer.
7. If no programs match at all, return an empty matches array.\
"""

RANKING_PROMPT_TEMPLATE = """\
User Profile:
- Country: {country_name} ({country})
- Situation: {situation}
- Need: {need}

Here are the document chunks from the knowledge base:

{chunks_text}

Based on this user's profile, identify the most relevant government programs.
Return JSON in this exact format:
{{
    "matches": [
        {{
            "program_name": "Name of the program",
            "description": "Simple 1-sentence description",
            "who_qualifies": "Who is eligible, in simple terms",
            "source_document": "Document title from the chunks",
            "relevance_score": 0.92
        }}
    ]
}}

Output ONLY valid JSON.\
"""


async def profile_match(
    country: str,
    situation: str,
    need: str,
) -> str:
    """Match a user profile against all documents in the knowledge base
    and return relevant government programs ranked by relevance.

    This powers the Proactive Eligibility Agent. Instead of waiting for
    the user to ask the right question, proactively recommends programs
    they may be eligible for.

    Args:
        country: Country code ("MY", "ID", "PH", "TH").
        situation: User's situation — one of: "worker", "business_owner",
                   "family", "disaster_victim", "unemployed", "student".
        need: Primary need — one of: "financial_aid", "healthcare",
              "worker_rights", "business_support", "housing", "legal_aid",
              "education".

    Returns:
        JSON string with type="recommendations", matches array, and profile_summary.
    """
    # ── Normalize inputs ──────────────────────────────────────
    country = country.strip().upper()
    situation = situation.strip().lower()
    need = need.strip().lower()
    country_name = COUNTRY_NAMES.get(country, country)

    profile_summary = (
        f"{situation.replace('_', ' ').title()} in {country_name} "
        f"seeking {need.replace('_', ' ')}"
    )

    # ── Step 1: Build search query from profile ───────────────
    need_phrase = NEED_PHRASES.get(need, need.replace("_", " "))
    query_template = QUERY_TEMPLATES.get(situation, "{need_phrase} government program benefits")
    query = query_template.format(need_phrase=need_phrase)

    # Get topic filter
    topic = TOPIC_MAP.get((situation, need), "")

    logger.info(
        "profile_match: country=%s, situation=%s, need=%s, topic=%s, query='%s'",
        country, situation, need, topic, query[:80],
    )

    # ── Step 2: Query ChromaDB ────────────────────────────────
    from db import search as chroma_search

    try:
        # Retrieve top 10 chunks with profile-based filters
        results = chroma_search(
            query=query,
            n_results=10,
            country=country,
            topic=topic,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

    except Exception as e:
        logger.error("ChromaDB search failed in profile_match: %s", e)
        return json.dumps({
            "type": "recommendations",
            "matches": [],
            "profile_summary": profile_summary,
            "total_matches": 0,
            "error": f"Search failed: {str(e)}",
        })

    if not documents:
        # If no results with topic filter, try without it
        logger.info("No results with topic filter, retrying without topic...")
        try:
            results = chroma_search(
                query=query,
                n_results=10,
                country=country,
                topic="",  # No topic filter
            )
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
        except Exception:
            pass

    if not documents:
        return json.dumps({
            "type": "recommendations",
            "matches": [],
            "profile_summary": profile_summary,
            "total_matches": 0,
            "note": "No matching documents found in the knowledge base for this profile.",
        })

    # ── Step 3: Build chunks text for LLM ranking ─────────────
    chunks_text = _format_chunks_for_llm(documents, metadatas, distances)

    # ── Step 4: LLM-based ranking ─────────────────────────────
    from llm_client import call_llm, LLMError

    ranking_prompt = RANKING_PROMPT_TEMPLATE.format(
        country_name=country_name,
        country=country,
        situation=situation.replace("_", " "),
        need=need.replace("_", " "),
        chunks_text=chunks_text,
    )

    try:
        raw = await call_llm(
            ranking_prompt,
            system_prompt=RANKING_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=2048,
            json_mode=True,
        )

        ranked = _parse_ranking_response(raw, country, need=need)

        result = {
            "type": "recommendations",
            "items": ranked[:4],  # Cap at 4 recommendations (frontend expects "items")
            "matches": ranked[:4],  # Keep for backwards compat
            "profile_summary": profile_summary,
            "total_matches": len(ranked),
        }

        logger.info(
            "profile_match: found %d matches for %s",
            len(ranked), profile_summary,
        )

        return json.dumps(result, ensure_ascii=False)

    except LLMError as e:
        logger.error("LLM ranking failed in profile_match: %s", e)
        # Fallback: return raw ChromaDB results as basic cards
        fallback = _fallback_matches(documents, metadatas, distances, country, need=need)
        return json.dumps({
            "type": "recommendations",
            "items": fallback,
            "matches": fallback,
            "profile_summary": profile_summary,
            "total_matches": min(4, len(documents)),
            "note": "Ranked by vector similarity (LLM ranking unavailable)",
        }, ensure_ascii=False)


def _format_chunks_for_llm(
    documents: list[str],
    metadatas: list[dict],
    distances: list[float],
) -> str:
    """Format ChromaDB results into text for the LLM ranking prompt."""
    parts = []
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        similarity = round(1 - (dist / 2), 3)
        title = meta.get("document_title", "Unknown")
        agency = meta.get("source_agency", "")
        topic = meta.get("topic", "")
        text_preview = doc[:500]  # Cap chunk length for prompt

        parts.append(
            f"[Chunk {i+1}] (similarity={similarity}, source={title}"
            f"{f', agency={agency}' if agency else ''}"
            f"{f', topic={topic}' if topic else ''})\n"
            f"{text_preview}\n"
        )

    return "\n".join(parts)


def _parse_ranking_response(raw: str, country: str, need: str = "") -> list[dict]:
    """Parse the LLM's JSON ranking response."""
    import re

    cleaned = raw.strip()

    # Try direct parse
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try extracting from code fences
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                return []
        else:
            # Try finding a JSON object
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    return []
            else:
                return []

    matches = data.get("matches", [])

    # Default icon based on need category
    default_icon = NEED_ICONS.get(need, "📋")

    # Ensure each match has the required fields (matching frontend Recommendation type)
    validated = []
    for i, m in enumerate(matches):
        if not isinstance(m, dict):
            continue
        validated.append({
            "id": f"rec_{i}",
            "title": m.get("program_name", "Unknown Program"),
            "program_name": m.get("program_name", "Unknown Program"),
            "description": m.get("description", ""),
            "eligibility": m.get("who_qualifies", ""),
            "who_qualifies": m.get("who_qualifies", ""),
            "source_document": m.get("source_document", ""),
            "country": country,
            "icon": default_icon,
            "relevance_score": min(1.0, max(0.0, float(m.get("relevance_score", 0.5)))),
        })

    # Sort by relevance score descending
    validated.sort(key=lambda x: x["relevance_score"], reverse=True)
    return validated


def _fallback_matches(
    documents: list[str],
    metadatas: list[dict],
    distances: list[float],
    country: str,
    need: str = "",
) -> list[dict]:
    """Build basic recommendation cards from raw ChromaDB results (no LLM)."""
    default_icon = NEED_ICONS.get(need, "📋")
    matches = []
    for i, (doc, meta, dist) in enumerate(zip(documents[:4], metadatas[:4], distances[:4])):
        similarity = round(1 - (dist / 2), 3)
        title = meta.get("document_title", "Government Program")
        matches.append({
            "id": f"rec_{i}",
            "title": title,
            "program_name": title,
            "description": doc[:150] + "..." if len(doc) > 150 else doc,
            "eligibility": "See document for eligibility details",
            "who_qualifies": "See document for eligibility details",
            "source_document": title,
            "country": country,
            "icon": default_icon,
            "relevance_score": round(similarity, 2),
        })
    return matches