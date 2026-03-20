"""
translate_text — Translate between SEA languages using LLM.

Owner: Pharthiban (tool code), Lineysha (prompt tuning)
Depends on: llm_client.py (async call_llm)

Approach:
  - Path A (primary): LLM-based translation via SEA-LION.
    Qwen-SEA-LION is trained on SEA languages and handles BM, ID, TL, TH natively.
  - No NLLB-200 offline fallback for now (saves dependency weight for hackathon).
  - Same-language requests short-circuit without an API call.
  - Prompt is tuned to preserve simplified language level (no jargon re-introduction).

Note: This function is async because it calls the async call_llm().
      The MCP server wrapper in mcp_server.py must also be async.
"""

import json
import logging

logger = logging.getLogger("askara.translate")

# ── Supported languages ──────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "ms": "Malay",
    "en": "English",
    "id": "Indonesian",
    "tl": "Filipino",
    "th": "Thai",
    "jv": "Javanese",
    "ceb": "Cebuano",
    "war": "Waray",
}

# Language pairs where the model is known to perform best
# (SEA-LION was trained heavily on these)
HIGH_QUALITY_PAIRS = {
    ("ms", "en"), ("en", "ms"),
    ("id", "en"), ("en", "id"),
    ("ms", "id"), ("id", "ms"),  # Closely related, very reliable
    ("tl", "en"), ("en", "tl"),
    ("th", "en"), ("en", "th"),
}
# ──────────────────────────────────────────────────────────────

# Translation system prompt — keeps output simple, no jargon
TRANSLATION_SYSTEM_PROMPT = """\
You are a professional translator for Southeast Asian government services.

Rules:
1. Translate accurately — do NOT add, remove, or change the meaning.
2. Keep the language SIMPLE. If the source text uses easy words, the translation must also use easy words.
3. Do NOT re-introduce jargon or formal terms. If the source says "work accident insurance", translate that phrase simply — do NOT convert it back to a technical term.
4. Keep sentences short and clear.
5. Preserve any names, reference numbers, dates, and amounts exactly as they are.
6. Output ONLY the translated text — no explanations, no notes, no preamble.\
"""


async def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
) -> str:
    """Translate text between SEA languages.

    Use this tool when the user's preferred language is different from the
    language of the retrieved documents. For example, an Indonesian worker
    asking about Malaysian SOCSO benefits — retrieve in Malay, translate
    the simplified result to Indonesian.

    Args:
        text: The text to translate (already simplified if possible).
        source_lang: Source language code ("ms", "en", "id", "tl", "th", "jv", "ceb", "war").
        target_lang: Target language code (same options as source_lang).

    Returns:
        JSON string:
        {
            "translated_text": "...",
            "source_lang": "ms",
            "target_lang": "id",
            "model_used": "sealion",
            "quality_note": "high" | "medium" | "best_effort"
        }
    """
    # ── Validation ────────────────────────────────────────────
    if not text or not text.strip():
        return json.dumps({
            "translated_text": "",
            "source_lang": source_lang,
            "target_lang": target_lang,
            "model_used": "none",
            "quality_note": "empty_input",
        })

    src = source_lang.strip().lower()
    tgt = target_lang.strip().lower()

    if src not in SUPPORTED_LANGUAGES:
        return json.dumps({
            "translated_text": text,
            "source_lang": src,
            "target_lang": tgt,
            "model_used": "none",
            "quality_note": f"unsupported_source_lang: {src}",
        })

    if tgt not in SUPPORTED_LANGUAGES:
        return json.dumps({
            "translated_text": text,
            "source_lang": src,
            "target_lang": tgt,
            "model_used": "none",
            "quality_note": f"unsupported_target_lang: {tgt}",
        })

    # ── Same-language shortcut ────────────────────────────────
    if src == tgt:
        return json.dumps({
            "translated_text": text,
            "source_lang": src,
            "target_lang": tgt,
            "model_used": "passthrough",
            "quality_note": "same_language",
        })

    # ── Build translation prompt ──────────────────────────────
    src_name = SUPPORTED_LANGUAGES[src]
    tgt_name = SUPPORTED_LANGUAGES[tgt]

    # Quality estimate
    pair = (src, tgt)
    if pair in HIGH_QUALITY_PAIRS:
        quality_note = "high"
    elif src in ("ms", "en", "id", "tl", "th") and tgt in ("ms", "en", "id", "tl", "th"):
        quality_note = "medium"
    else:
        quality_note = "best_effort"

    prompt = (
        f"Translate the following text from {src_name} to {tgt_name}.\n\n"
        f"Text:\n{text}"
    )

    # ── Call LLM ──────────────────────────────────────────────
    from llm_client import call_llm, LLMError

    model_used = "sealion"

    try:
        translated = await call_llm(
            prompt,
            system_prompt=TRANSLATION_SYSTEM_PROMPT,
            temperature=0.2,  # Low temp for faithful translation
            max_tokens=2048,
        )

        # Clean up common LLM artifacts
        translated = _clean_translation(translated, text)

        logger.info(
            "translate: %s→%s, input=%d chars, output=%d chars, quality=%s",
            src, tgt, len(text), len(translated), quality_note,
        )

        return json.dumps({
            "translated_text": translated,
            "source_lang": src,
            "target_lang": tgt,
            "model_used": model_used,
            "quality_note": quality_note,
        }, ensure_ascii=False)

    except LLMError as e:
        logger.error("Translation LLM call failed: %s", e)
        return json.dumps({
            "translated_text": text,  # Return original as fallback
            "source_lang": src,
            "target_lang": tgt,
            "model_used": "fallback_passthrough",
            "quality_note": f"llm_error: {str(e)}",
        }, ensure_ascii=False)


def _clean_translation(translated: str, original: str) -> str:
    """Clean up common LLM translation artifacts.

    Models sometimes prepend "Translation:" or wrap in quotes.
    """
    cleaned = translated.strip()

    # Remove common prefixes the model sometimes adds
    prefixes_to_strip = [
        "Translation:", "Translated text:", "Here is the translation:",
        "Terjemahan:", "Berikut terjemahannya:",
    ]
    for prefix in prefixes_to_strip:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()

    # Remove wrapping quotes if the original didn't have them
    if (cleaned.startswith('"') and cleaned.endswith('"') and
            not original.startswith('"')):
        cleaned = cleaned[1:-1].strip()

    return cleaned