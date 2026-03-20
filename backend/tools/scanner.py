"""
scan_document — OCR a photographed document image (Snap & Understand feature).

Owner: Pharthiban (Day 5)
Depends on: llm_client.py (call_llm_vision for VL, call_llm_json for structure analysis)

Pipeline:
  Primary: Send base64 image to Qwen-SEA-LION-v4-4B-VL via OpenAI-compatible vision API
    - 256K context, superior SEA OCR for ID/TH/VN documents
    - Single pass: extract text + identify structure in one call

  Fallback: pytesseract with lang packs → regular LLM for structure analysis
    - Used when VL model is unavailable or returns an error
    - Requires: pip install pytesseract Pillow, system tesseract-ocr installed

Note: This function is async because it calls async LLM functions.
"""

import json
import logging
import re
import base64
from io import BytesIO

logger = logging.getLogger("askara.scanner")

# ── VL model extraction prompt ───────────────────────────────
VL_EXTRACTION_PROMPT = """\
You are a document analysis expert for Southeast Asian government documents.

Analyze this document image and extract:

1. ALL text visible in the document, preserving the layout and sections.
2. Document type (letter, form, notice, receipt, statement, certificate, or unknown).
3. Issuing agency or organization name.
4. Language of the document (ms=Malay, id=Indonesian, en=English, tl=Filipino, th=Thai).
5. Key information:
   - Any dates mentioned
   - Any monetary amounts (with currency: RM, Rp, PHP, THB)
   - Any reference numbers, case IDs, or IC numbers

Respond in this exact JSON format:
{
    "extracted_text": "full text here with line breaks preserved",
    "document_type": "letter",
    "detected_language": "ms",
    "issuing_agency": "PERKESO",
    "key_info": {
        "dates": ["2024-03-15"],
        "amounts": ["RM 1,200.00"],
        "reference_numbers": ["SBP/2024/12345"]
    }
}

If you cannot read part of the text clearly, indicate it with [unclear].
Output ONLY valid JSON — no markdown fences, no explanations.\
"""

# ── Structure analysis prompt (for tesseract fallback) ───────
STRUCTURE_ANALYSIS_PROMPT = """\
You are a document analysis expert for Southeast Asian government documents.

I have extracted this text from a government document using OCR.
Analyze the text and identify:

1. Document type (letter, form, notice, receipt, statement, certificate, or unknown).
2. Issuing agency or organization.
3. Language of the document (ms, id, en, tl, th).
4. Key information: dates, monetary amounts, reference numbers.

OCR Extracted Text:
{text}

Respond in this exact JSON format:
{{
    "document_type": "letter",
    "detected_language": "ms",
    "issuing_agency": "PERKESO",
    "key_info": {{
        "dates": ["2024-03-15"],
        "amounts": ["RM 1,200.00"],
        "reference_numbers": ["SBP/2024/12345"]
    }}
}}

Output ONLY valid JSON.\
"""

# ── Tesseract language packs ─────────────────────────────────
TESSERACT_LANG_MAP = {
    "ms": "msa",      # Malay
    "id": "ind",      # Indonesian
    "en": "eng",      # English
    "tl": "fil",      # Filipino
    "th": "tha",      # Thai
}


async def scan_document(
    image_base64: str,
    source_hint: str = "",
) -> str:
    """OCR a photographed government document and extract structured text.

    This powers the "Snap & Understand" feature. The user photographs
    a letter from PERKESO, a JKM flood relief notice, a BPJS statement,
    etc., and this tool extracts the text so the agent can simplify and
    translate it.

    Args:
        image_base64: Base64-encoded image data (JPEG or PNG from camera).
        source_hint: Optional hint about the document ("PERKESO letter",
                     "JKM notice", etc.) to improve extraction accuracy.

    Returns:
        JSON string with extracted_text, document_type, detected_language,
        issuing_agency, key_info, confidence, and engine.
    """
    if not image_base64 or not image_base64.strip():
        return json.dumps({
            "extracted_text": "",
            "document_type": "unknown",
            "detected_language": "",
            "issuing_agency": "",
            "key_info": {"dates": [], "amounts": [], "reference_numbers": []},
            "confidence": 0.0,
            "engine": "none",
            "error": "No image data provided",
        })

    # Strip data URI prefix if present
    clean_b64 = image_base64
    if clean_b64.startswith("data:"):
        clean_b64 = clean_b64.split(",", 1)[1]

    # Detect image type from base64 header
    media_type = _detect_media_type(clean_b64)

    # ── Try VL model first ────────────────────────────────────
    try:
        result = await _scan_with_vl(clean_b64, source_hint, media_type)
        logger.info("scan_document: VL model succeeded")
        return result
    except Exception as e:
        logger.warning("VL model failed, falling back to tesseract: %s", e)

    # ── Fallback: Tesseract OCR + LLM analysis ───────────────
    try:
        result = await _scan_with_tesseract(clean_b64, source_hint)
        logger.info("scan_document: Tesseract fallback succeeded")
        return result
    except Exception as e:
        logger.error("Tesseract fallback also failed: %s", e)
        return json.dumps({
            "extracted_text": "",
            "document_type": "unknown",
            "detected_language": "",
            "issuing_agency": "",
            "key_info": {"dates": [], "amounts": [], "reference_numbers": []},
            "confidence": 0.0,
            "engine": "error",
            "error": f"Both VL and Tesseract failed: {str(e)}",
        })


async def _scan_with_vl(
    image_base64: str,
    source_hint: str,
    media_type: str = "image/jpeg",
) -> str:
    """Extract text using SEA-LION VL model (primary path)."""
    from llm_client import call_llm_vision, LLMError

    prompt = VL_EXTRACTION_PROMPT
    if source_hint:
        prompt += f"\n\nHint: This appears to be a {source_hint}."

    raw = await call_llm_vision(
        image_base64=image_base64,
        prompt=prompt,
        temperature=0.1,
        max_tokens=4096,
        image_media_type=media_type,
    )

    # Parse the JSON response
    parsed = _parse_json_response(raw)

    # Build standardized result
    result = {
        "extracted_text": parsed.get("extracted_text", raw),
        "document_type": parsed.get("document_type", "unknown"),
        "detected_language": parsed.get("detected_language", ""),
        "issuing_agency": parsed.get("issuing_agency", ""),
        "key_info": parsed.get("key_info", {
            "dates": [],
            "amounts": [],
            "reference_numbers": [],
        }),
        "confidence": 0.85,  # VL model is generally high confidence
        "engine": "sealion_vl",
    }

    # If VL didn't return structured JSON, still salvage the text
    if not parsed.get("extracted_text"):
        result["extracted_text"] = raw
        result["confidence"] = 0.6

    return json.dumps(result, ensure_ascii=False)


async def _scan_with_tesseract(
    image_base64: str,
    source_hint: str,
) -> str:
    """Extract text using Tesseract OCR + LLM structure analysis (fallback)."""
    from llm_client import call_llm, LLMError

    # Import optional deps
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        raise RuntimeError(
            "Tesseract fallback requires: pip install pytesseract Pillow "
            "and system tesseract-ocr installed"
        )

    # Decode base64 → PIL Image
    img_bytes = base64.b64decode(image_base64)
    img = Image.open(BytesIO(img_bytes))

    # Try multilingual OCR (Malay + English + Indonesian)
    lang_str = "msa+eng+ind"
    try:
        extracted_text = pytesseract.image_to_string(img, lang=lang_str)
    except pytesseract.TesseractError:
        # Fallback to English-only if lang packs missing
        extracted_text = pytesseract.image_to_string(img, lang="eng")
        logger.warning("Tesseract lang packs not found, using English only")

    extracted_text = extracted_text.strip()

    if not extracted_text:
        return json.dumps({
            "extracted_text": "",
            "document_type": "unknown",
            "detected_language": "",
            "issuing_agency": "",
            "key_info": {"dates": [], "amounts": [], "reference_numbers": []},
            "confidence": 0.1,
            "engine": "tesseract",
            "error": "Tesseract could not extract any text from the image",
        })

    # Use LLM to analyze the extracted text structure
    analysis_prompt = STRUCTURE_ANALYSIS_PROMPT.format(text=extracted_text[:3000])

    try:
        raw = await call_llm(
            analysis_prompt,
            temperature=0.1,
            max_tokens=1024,
            json_mode=True,
        )
        parsed = _parse_json_response(raw)
    except LLMError:
        # If LLM fails, return text-only result
        parsed = {}

    result = {
        "extracted_text": extracted_text,
        "document_type": parsed.get("document_type", "unknown"),
        "detected_language": parsed.get("detected_language", ""),
        "issuing_agency": parsed.get("issuing_agency", ""),
        "key_info": parsed.get("key_info", {
            "dates": _extract_dates(extracted_text),
            "amounts": _extract_amounts(extracted_text),
            "reference_numbers": _extract_references(extracted_text),
        }),
        "confidence": 0.55,  # Tesseract + LLM is lower confidence
        "engine": "tesseract",
    }

    return json.dumps(result, ensure_ascii=False)


# ── Helpers ──────────────────────────────────────────────────

def _detect_media_type(b64_data: str) -> str:
    """Detect image MIME type from base64 data header bytes."""
    try:
        header = base64.b64decode(b64_data[:32])
        if header[:3] == b'\xff\xd8\xff':
            return "image/jpeg"
        elif header[:8] == b'\x89PNG\r\n\x1a\n':
            return "image/png"
        elif header[:4] == b'RIFF' and header[8:12] == b'WEBP':
            return "image/webp"
    except Exception:
        pass
    return "image/jpeg"  # Default assumption


def _parse_json_response(raw: str) -> dict:
    """Try to parse JSON from LLM response, handling common artifacts."""
    cleaned = raw.strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding a JSON object in the text
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def _extract_dates(text: str) -> list[str]:
    """Extract date-like patterns from text (regex fallback)."""
    patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',      # DD/MM/YYYY, DD-MM-YYYY
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',          # YYYY-MM-DD
        r'\d{1,2}\s+\w+\s+\d{4}',                 # 15 Mac 2024, 1 Januari 2025
    ]
    dates = []
    for pattern in patterns:
        dates.extend(re.findall(pattern, text))
    return dates[:10]  # Cap at 10


def _extract_amounts(text: str) -> list[str]:
    """Extract monetary amounts from text (regex fallback)."""
    patterns = [
        r'RM\s*[\d,]+\.?\d*',                     # RM 1,200.00
        r'Rp\.?\s*[\d,.]+',                        # Rp. 500.000
        r'PHP\s*[\d,]+\.?\d*',                     # PHP 5,000.00
        r'THB\s*[\d,]+\.?\d*',                     # THB 3,000
        r'\$\s*[\d,]+\.?\d*',                      # $500.00
    ]
    amounts = []
    for pattern in patterns:
        amounts.extend(re.findall(pattern, text))
    return amounts[:10]


def _extract_references(text: str) -> list[str]:
    """Extract reference number patterns from text (regex fallback)."""
    patterns = [
        r'[A-Z]{2,5}[/-]\d{4}[/-]\d{3,10}',      # SBP/2024/12345
        r'No\.\s*Rujukan[:\s]+\S+',                # No. Rujukan: ABC123
        r'Ref[:\s]+\S+',                           # Ref: ABC123
        r'\b\d{6,14}\b',                           # Long number (IC, reference)
    ]
    refs = []
    for pattern in patterns:
        refs.extend(re.findall(pattern, text))
    return refs[:10]