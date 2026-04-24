"""
summarize_text — Summarization into actionable Step Cards or bullets.

Owner: Lineysha (NLP logic), Pharthiban (wiring)
Depends on: llm_client.py

Pipeline:
1. If text > 500 words → chunk by paragraphs → summarize each → combine
2. Final LLM pass to format as step_cards or bullets
3. Return structured JSON matching frontend schema
"""

import json
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm_client import call_llm, call_llm_json, LLMError

logger = logging.getLogger("askara.summarize")


# ── Step Card schema (frontend contract) ──────────────────────
# The frontend StepCards.tsx component expects this exact shape.
# DO NOT change the schema without updating the frontend.
#
# {
#     "type": "step_cards",
#     "summary": "Brief intro text (optional)",
#     "cards": [
#         {
#             "step": 1,
#             "total": 3,
#             "title": "Short action title",
#             "icon": "📋",                        (optional emoji)
#             "body": "What to do, explained simply",
#             "location": "Office name + address",  (optional)
#             "hours": "Mon-Fri, 8:30am-4:30pm",   (optional)
#             "deadline": "30 March 2026",          (optional)
#             "amount": "RM500",                    (optional)
#             "checklist": ["Item 1", "Item 2"],    (optional)
#             "action": {
#                 "type": "link" | "call" | "navigate" | "share" | "none",
#                 "label": "Button text",
#                 "url": "https://...",             (for link/share)
#                 "phone": "1800228000",            (for call)
#                 "lat": 6.12,                      (for navigate)
#                 "lng": 102.24                     (for navigate)
#             }
#         }
#     ]
# }
# ──────────────────────────────────────────────────────────────


# ── Prompts ───────────────────────────────────────────────────

STEP_CARDS_SYSTEM_PROMPT = """You are a practical government services guide for AskAra+, helping ASEAN migrant workers and \
vulnerable populations navigate Malaysian government processes.

Your task: Generate {max_steps} ACTIONABLE step cards in {language} that walk the user \
through the COMPLETE process from start to finish.

## CRITICAL: GENERATE A COMPLETE GUIDE

The provided text may only cover PART of the process (e.g. only eligibility rules, or only \
one legal clause). You MUST generate a FULL step-by-step guide anyway — from the very first \
action the user must take all the way to receiving the certificate/approval/benefit.

Use the provided text as your PRIMARY source for specific details. \
For steps that the text does not cover, use your accurate knowledge of \
standard Malaysian government procedures to fill in the remaining steps.

AIM: Generate between 3 and {max_steps} steps that cover the full process end to end.

---

## TWO TIERS OF INFORMATION

### TIER 1 — TEXT-GROUNDED (extract ONLY from the provided text):
- Specific RM/Rp fees and amounts
- Specific phone numbers and hotlines
- Specific URLs and online portal addresses
- Exact deadlines (e.g. "within 60 days of injury")
- Specific form names and codes
- Specific eligibility criteria and exemptions stated in the text

### TIER 2 — PROCESS-COMPLETE (use your knowledge of Malaysian procedures):
- Which office or government portal to visit (SSM, DBKL, LHDN, PERKESO, JTK, etc.)
- What standard documents to prepare (MyKad, passport-size photos, tenancy agreement, etc.)
- Correct sequence of steps (e.g. register name → get business cert → apply for license)
- Standard waiting times where commonly known

---

## CONTENT RULES:
1. TIER 1: NEVER invent specific amounts, phone numbers, URLs, or form codes not in the text.
2. TIER 2: DO generate correct process steps using your knowledge of Malaysian government procedures.
3. Each step must be SPECIFIC and ACTIONABLE — not vague.
   BAD: "Prepare your documents"
   GOOD: "Prepare your MyKad (IC), 2 passport-size photos, and proof of business address (e.g. tenancy agreement or utility bill)"
   BAD: "Visit the office"
   GOOD: "Go to the nearest SSM (Suruhanjaya Syarikat Malaysia) office or use the MySSM online portal to register your business name and get your registration number"
4. Eligibility or exemption info from the text goes in Step 1 as a quick eligibility check.
5. Put required documents in the checklist field. Put the specific office or portal in the location field.
6. The "total" field in EACH card must equal the total number of cards you generate.
7. Use simple language — Grade 5 reading level, short sentences, everyday words.

---

## EXTRACTION PRIORITIES FROM THE TEXT:
- Required documents or forms → checklist field
- Specific office name or portal URL → location field
- Operating hours → hours field
- Exact fee amounts (RM) → amount field
- Deadlines → deadline field
- Hotlines / phone numbers / websites → action field (type: "call" or "link")

Return ONLY valid JSON. No markdown, no preamble, no explanation.

JSON format:
{{
  "type": "step_cards",
  "summary": "Brief 1-sentence intro in {language} — what these steps help the user do",
  "cards": [
    {{
      "step": 1,
      "total": <total number of cards you generate>,
      "title": "Short action title (max 6 words)",
      "icon": "<one relevant emoji>",
      "body": "Clear, specific explanation of what the user does in this step",
      "location": "<office name or portal — from text or well-known Malaysian agency>",
      "hours": "<ONLY if stated in text>",
      "deadline": "<ONLY if stated in text>",
      "amount": "<ONLY if stated in text, e.g. RM60>",
      "checklist": ["<specific document or item>"],
      "action": {{
        "type": "link|call|none",
        "label": "Button label (e.g. Apply Online, Call Hotline)",
        "url": "<ONLY from text>",
        "phone": "<ONLY from text>"
      }}
    }}
  ]
}}
"""

BULLETS_SYSTEM_PROMPT = """\
You are a summarization assistant for AskAra+.

Extract the key information from the text into concise bullet points in {language}.
ONLY include facts explicitly stated in the text. NEVER invent information.

Return ONLY valid JSON:
{{"type": "bullets", "points": ["Point 1", "Point 2", "Point 3"]}}\
"""


# ── Main tool function ────────────────────────────────────────

async def summarize_text(
    text: str,
    format: str = "step_cards",
    language: str = "en",
    max_steps: int = 5,
) -> str:
    """Summarize government text into Step Cards or bullet points.

    Args:
        text: The text to summarize (ideally already simplified).
        format: "step_cards" (default, for procedures) or "bullets" (for info).
        language: Output language code ("en", "ms", "id", "tl", "th").
        max_steps: Maximum number of step cards (default 5).

    Returns:
        JSON string matching the frontend schema.
    """
    logger.info(
        "[summarize] format=%s, language=%s, input_words=%d",
        format, language, len(text.split()),
    )

    # ── Step 1: Chunk long texts ──────────────────────────────
    # Skip chunking when the context is already pre-formatted by the agent
    # (detected by the "USER QUESTION:" header).  Chunking a structured
    # context burns extra LLM calls and fragments the intentional structure.
    is_preformatted = text.lstrip().startswith("USER QUESTION:")

    if not is_preformatted and len(text.split()) > 500:
        logger.info("[summarize] Text > 500 words — chunking paragraphs")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        summaries = []
        for p in paragraphs:
            try:
                summary = await call_llm(
                    f"Summarize this text briefly in 2-3 sentences. "
                    f"Keep all important facts, numbers, and names:\n\n{p}",
                    temperature=0.1,
                )
                summaries.append(summary)
            except LLMError as e:
                logger.warning("[summarize] Chunk summarization failed: %s", e)
                # Keep original paragraph as fallback
                summaries.append(p[:300])

        text = "\n\n".join(summaries)
    elif is_preformatted:
        logger.info("[summarize] Pre-formatted context detected — skipping chunk step")

    # ── Step 2: Format into structured output ─────────────────
    try:
        if format == "step_cards":
            system_prompt = STEP_CARDS_SYSTEM_PROMPT.format(
                max_steps=max_steps,
                language=language,
            )

            result_dict = await call_llm_json(
                prompt=f"Text to convert into step cards:\n\n{text}",
                system_prompt=system_prompt,
            )

            # Validate and clean the result
            result_dict = _validate_step_cards(result_dict, max_steps)

        else:
            system_prompt = BULLETS_SYSTEM_PROMPT.format(language=language)

            result_dict = await call_llm_json(
                prompt=f"Text to extract bullet points from:\n\n{text}",
                system_prompt=system_prompt,
            )

            # Validate bullets
            result_dict = _validate_bullets(result_dict)

        logger.info(
            "[summarize] Success — type=%s, items=%d",
            result_dict.get("type"),
            len(result_dict.get("cards", result_dict.get("points", []))),
        )

        return json.dumps(result_dict, ensure_ascii=False)

    except (LLMError, Exception) as e:
        logger.error("[summarize] LLM call failed: %s — returning fallback", e)
        return _fallback_response(text, format, language)


# ── Validation helpers ────────────────────────────────────────

def _normalize_url(url: str) -> str:
    """Ensure a URL always has an https:// scheme.

    The LLM frequently omits the scheme (e.g. "www.perkeso.gov.my").
    Without a scheme, browsers treat the href as a relative path and
    prepend the current origin → http://localhost:3000/www.perkeso.gov.my.
    """
    if not url or not isinstance(url, str):
        return url
    url = url.strip()
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return f"https://{url}"


def _validate_step_cards(data: dict, max_steps: int) -> dict:
    """Ensure step_cards output matches the frontend schema."""
    if not isinstance(data, dict):
        data = {}

    data["type"] = "step_cards"

    if "summary" not in data or not isinstance(data.get("summary"), str):
        data["summary"] = ""

    cards = data.get("cards", [])
    if not isinstance(cards, list):
        cards = []

    # Cap at max_steps
    cards = cards[:max_steps]

    # Fix step numbering and totals
    total = len(cards)
    validated_cards = []

    for i, card in enumerate(cards):
        if not isinstance(card, dict):
            continue

        clean_card = {
            "step": i + 1,
            "total": total,
            "title": card.get("title", f"Step {i + 1}"),
            "body": card.get("body", ""),
        }

        # Only include optional fields if they have real values
        # (not empty strings, not placeholder text)
        for field in ["icon", "location", "hours", "deadline", "amount"]:
            value = card.get(field)
            if value and isinstance(value, str) and value.strip():
                # Filter out obvious placeholder patterns
                if not _is_placeholder(value):
                    clean_card[field] = value.strip()

        # Checklist
        checklist = card.get("checklist")
        if isinstance(checklist, list) and len(checklist) > 0:
            clean_card["checklist"] = [
                str(item) for item in checklist
                if item and str(item).strip()
            ]

        # Action
        action = card.get("action")
        if isinstance(action, dict) and action.get("type"):
            action_type = action["type"]
            if action_type in ("link", "call", "navigate", "share", "none"):
                clean_action = {"type": action_type}
                if action.get("label"):
                    clean_action["label"] = action["label"]
                if action_type == "link" and action.get("url"):
                    clean_action["url"] = _normalize_url(action["url"])
                elif action_type == "share" and action.get("url"):
                    clean_action["url"] = _normalize_url(action["url"])
                elif action_type == "call" and action.get("phone"):
                    clean_action["phone"] = action["phone"]
                elif action_type == "navigate":
                    if action.get("lat") and action.get("lng"):
                        clean_action["lat"] = action["lat"]
                        clean_action["lng"] = action["lng"]
                    elif action.get("url"):
                        clean_action["url"] = _normalize_url(action["url"])
                clean_card["action"] = clean_action

        validated_cards.append(clean_card)

    data["cards"] = validated_cards
    return data


def _validate_bullets(data: dict) -> dict:
    """Ensure bullets output matches the frontend schema."""
    if not isinstance(data, dict):
        data = {}

    data["type"] = "bullets"

    points = data.get("points", [])
    if not isinstance(points, list):
        points = []

    data["points"] = [str(p) for p in points if p and str(p).strip()]
    return data


def _is_placeholder(value: str) -> bool:
    """Detect obvious placeholder/hallucinated values."""
    placeholders = [
        "office name",
        "address",
        "no. 123",
        "jalan merdeka",
        "https://...",
        "1800228000",
        "button text",
        "item 1",
        "item 2",
        "point 1",
        "point 2",
        "rm0",
        "none",
    ]
    lower = value.lower().strip()
    return any(p in lower for p in placeholders)


# ── Fallback response ─────────────────────────────────────────

def _fallback_response(text: str, format: str, language: str) -> str:
    """Return a basic response when LLM fails."""
    if format == "step_cards":
        # Extract first few sentences as a single card
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        body = ". ".join(sentences[:3]) + "." if sentences else text[:200]

        result = {
            "type": "step_cards",
            "summary": "Here is the information we found:",
            "cards": [
                {
                    "step": 1,
                    "total": 1,
                    "title": "Information",
                    "icon": "📋",
                    "body": body,
                    "action": {"type": "none"},
                }
            ],
        }
    else:
        sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
        result = {
            "type": "bullets",
            "points": sentences[:5] if sentences else [text[:200]],
        }

    return json.dumps(result, ensure_ascii=False)


# ── Standalone test ───────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    test_text = (
        "Pekerja yang layak boleh memohon Skim Bencana Pekerjaan di bawah PERKESO. "
        "Caruman bulanan perlu dibayar oleh majikan. "
        "Permohonan hendaklah dikemukakan dalam tempoh 60 hari dari tarikh kemalangan. "
        "Bawa MyKad asal dan salinan, kontrak pekerjaan, dan surat doktor."
    )

    result = asyncio.run(summarize_text(test_text, language="ms"))
    parsed = json.loads(result)
    print(json.dumps(parsed, indent=2, ensure_ascii=False))