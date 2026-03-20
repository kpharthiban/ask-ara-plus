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

STEP_CARDS_SYSTEM_PROMPT = """\
You are a summarization assistant for AskAra+, an ASEAN government services helper.

Your task: Convert the provided text into {max_steps} or fewer actionable step cards in {language}.

EXTRACTION PRIORITIES — look for and INCLUDE these in the cards:
1. REQUIRED DOCUMENTS: What forms, IDs, letters does the user need? List them in the checklist field.
2. WHERE TO GO: Which specific office, portal, or counter? Put in the location field.
3. ELIGIBILITY: Who qualifies? State this clearly in the body of the first card.
4. AMOUNTS & DEADLINES: Any RM/Rp amounts, time limits (e.g. "within 60 days")? Put in amount/deadline fields.
5. CONTACT INFO: Any hotline, phone number, or website? Use the action field with type "call" or "link".
6. SPECIFIC STEPS: What exactly does the user DO at each stage? Not "go to office" — say "go to [specific office name] and submit [specific form]".

CRITICAL RULES:
1. ONLY use information explicitly stated in the provided text.
2. NEVER invent addresses, phone numbers, office names, amounts, deadlines, or URLs.
3. Each step must be SPECIFIC and ACTIONABLE — not vague advice like "read guidelines" or "prepare documents".
   - BAD: "Prepare your documents"
   - GOOD: "Bring your MyKad (IC), payslips for the last 3 months, and employer letter"
   - BAD: "Visit the office"
   - GOOD: "Go to the nearest PERKESO office or apply online at perkeso.gov.my"
4. If the text mentions specific forms, documents, or requirements, put them in the "checklist" array.
5. Only include optional fields (location, hours, deadline, amount, checklist, action) if the text provides real data for them. Omit entirely if no data.
6. The "total" field in each card must equal the total number of cards.
7. Use simple language (Grade 5 reading level).

Return ONLY valid JSON. No markdown, no explanation, no preamble.

JSON format:
{{
  "type": "step_cards",
  "summary": "Brief 1-sentence intro in {language}",
  "cards": [
    {{
      "step": 1,
      "total": <total number of cards>,
      "title": "Short action title (max 6 words)",
      "icon": "<relevant emoji>",
      "body": "Clear, specific explanation of what to do",
      "location": "<ONLY if mentioned in text>",
      "hours": "<ONLY if mentioned in text>",
      "deadline": "<ONLY if mentioned in text>",
      "amount": "<ONLY if mentioned in text>",
      "checklist": ["<specific items from text>"],
      "action": {{
        "type": "link|call|none",
        "label": "Button text",
        "url": "<if link type, from text>",
        "phone": "<if call type, from text>"
      }}
    }}
  ]
}}\
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
    if len(text.split()) > 500:
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
                    clean_action["url"] = action["url"]
                elif action_type == "call" and action.get("phone"):
                    clean_action["phone"] = action["phone"]
                elif action_type == "navigate":
                    if action.get("lat") and action.get("lng"):
                        clean_action["lat"] = action["lat"]
                        clean_action["lng"] = action["lng"]
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