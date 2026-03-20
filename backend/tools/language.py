"""
detect_language() — Real implementation (Option A: Fast)
Uses langdetect for primary language + regex for code-switching detection.
Zero LLM calls — instant response. Can upgrade to LLM-based (Option B) later.
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from langdetect import detect, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not installed — falling back to basic heuristics")


# ── ISO mapping: langdetect codes → our metadata codes ──
LANGDETECT_TO_ISO = {
    "ms": "ms",
    "id": "id",
    "tl": "tl",
    "th": "th",
    "en": "en",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "ta": "ta",
    "hi": "hi",
    "jw": "jv",    # Javanese (langdetect uses "jw")
    "vi": "vi",
    "my": "my",    # Burmese
    "km": "km",
    "lo": "lo",
}

# ── Language → likely country mapping ──
LANG_TO_COUNTRY_HINT = {
    "ms": "MY",
    "id": "ID",
    "tl": "PH",
    "th": "TH",
    "en": "",
    "jv": "ID",
    "ta": "MY",
}

# ── Code-switching markers ──

# English words commonly code-switched into SEA languages
ENGLISH_MARKERS = re.compile(
    r'\b(apply|register|check|online|download|form|status|payment|'
    r'submit|cancel|update|account|login|password|email|website|'
    r'office|document|card|number|already|actually|so|but|because|'
    r'confirm|approve|reject|pending|eligible|benefit|claim|'
    r'appointment|deadline|requirement|process|income)\b',
    re.IGNORECASE
)

# FIX: Malay-specific markers — removed shared words (saya, bantuan, permohonan, buat, ikut)
# Only words that are distinctly Malay, NOT Indonesian
MALAY_MARKERS = re.compile(
    r'\b(nak|tak|macam\s*mana|pejabat|caruman|'
    r'kerajaan|kena|kat|dah|lah|'
    r'tu|ni|je|kot|kan|eh|leh|'
    r'boleh tak|tak boleh|macam|bagi|'
    r'awak|encik|cik|puan|ringgit|MyKad|KWSP|PERKESO)\b',
    re.IGNORECASE
)

# FIX: Indonesian-specific markers — removed shared words
# Only words that are distinctly Indonesian, NOT Malay
INDONESIAN_MARKERS = re.compile(
    r'\b(tidak|bagaimana|bisa|kantor|iuran|'
    r'pemerintah|harus|sudah|'
    r'dong|sih|kok|lho|nih|gue|gak|nggak|gimana|udah|'
    r'enggak|nanya|belum|sama|kayak|banget|'
    r'kecamatan|kelurahan|puskesmas|BPJS|KTP|NIK)\b',
    re.IGNORECASE
)

# Thai script detection
THAI_SCRIPT = re.compile(r'[\u0E00-\u0E7F]')

# Filipino markers
FILIPINO_MARKERS = re.compile(
    r'\b(ang|ng|mga|ko|namin|gusto|hindi|alam|mag-|nag-|pag-|'
    r'po|opo|paano|saan|kailan|sino|bakit|kailangan|'
    r'puwede|ayuda|benepisyo)\b',
    re.IGNORECASE
)

# Dialect markers
KELANTAN_MARKERS = re.compile(
    r'\b(ambo|demo|gapo|gini|kito|make|mugo|nok|oghe|pah|'
    r'sapa|tok|wak|abe|che|nate|pitih|ghoyak|mari|tubik|'
    r'mano|dok|kecek|maghi|gak|golek|hungga|gedebe|'
    r'mung|buleh|dop|ore|doh|lum|leh|sino|bile|dio|'
    r'gapok|satgi|sekaroh|tanyo|rhaso|maknga)\b',
    re.IGNORECASE
)

JAVANESE_MARKERS = re.compile(
    r'\b(aku|kowe|opo|piye|sopo|wis|ora|iso|arep|wong|'
    r'kanggo|saking|nggih|inggih|dalem|mboten|panjenengan|'
    r'lungo|teko|ngendi|ngopo|emoh|rapopo|turu|tangi|'
    r'mangan|ngombe|ngomong|ndelok|krungu|njaluk|njupuk|'
    r'tuku|gowo|ngerti|lali|eling|gelem|gede|'
    r'cilik|akeh|sithik|apik|elek|durung|deweke|awakdewe)\b',
    re.IGNORECASE
)

WARAY_MARKERS = re.compile(
    r'\b(waray|diri|hin|han|nga|hini|adto|tikang|'
    r'pinakaharani|baga|bulig|sadto|ngan|sugad|kay|'
    r'hadto|sulod|gawas|panginabuhi|papeles|katungod|'
    r'pagsalbar|trabahador|kahimtangan|paghatag)\b',
    re.IGNORECASE
)

KHAM_MUEANG_MARKERS = re.compile(
    r'(เฮา|หื้อ|ก๋าน|ก๋อม|จ้วย|เบิ่ง|แอ่ว|'
    r'พ่อเมือง|คนเมือง|อู้|บ่า|ตั๋ว|แหม|เตื่อ|ม่วน|'
    r'ก๋ด|หนา|ของเฮา|บ้านเฮา)',
    re.IGNORECASE
)


def detect_language(text: str) -> str:
    """
    Detect the primary language, dialect, and any code-mixing in the input text.

    Returns:
        JSON string with primary_lang, dialect, secondary_langs,
        is_code_mixed, confidence, country_hint
    """
    if not text or not text.strip():
        return json.dumps({
            "primary_lang": "en",
            "dialect": "standard",
            "secondary_langs": [],
            "is_code_mixed": False,
            "confidence": 0.0,
            "country_hint": "",
        })

    text = text.strip()

    # FIX: Check for dialect FIRST (before langdetect can misclassify)
    # Kelantan text is often misclassified as Indonesian by langdetect
    dialect = _detect_dialect_early(text)

    # Step 1: Primary language detection via langdetect
    primary_lang, confidence = _detect_primary(text)

    # Step 2: Detect secondary languages (code-switching)
    secondary_langs = _detect_secondary_languages(text, primary_lang)
    is_code_mixed = len(secondary_langs) > 0

    # Step 3: Refine Malay vs Indonesian (langdetect often confuses them)
    if primary_lang in ("ms", "id"):
        primary_lang = _refine_malay_indonesian(text, primary_lang)

    # FIX: If dialect is Kelantan, force primary language to Malay
    if dialect == "kelantan":
        primary_lang = "ms"

    # Step 4: If no early dialect found, check again with corrected primary
    if dialect == "standard":
        dialect = _detect_dialect(text, primary_lang)

    # Step 5: Country hint
    country_hint = LANG_TO_COUNTRY_HINT.get(primary_lang, "")

    return json.dumps({
        "primary_lang": primary_lang,
        "dialect": dialect,
        "secondary_langs": secondary_langs,
        "is_code_mixed": is_code_mixed,
        "confidence": round(confidence, 4),
        "country_hint": country_hint,
    })


def _detect_primary(text: str) -> tuple[str, float]:
    """Use langdetect to get primary language + confidence."""
    if not LANGDETECT_AVAILABLE:
        return _fallback_detect(text)

    try:
        langs = detect_langs(text)
        if not langs:
            return ("en", 0.5)

        top = langs[0]
        lang_code = LANGDETECT_TO_ISO.get(str(top.lang), str(top.lang))
        return (lang_code, top.prob)

    except LangDetectException:
        return _fallback_detect(text)


def _fallback_detect(text: str) -> tuple[str, float]:
    """Basic heuristic fallback if langdetect is unavailable."""
    if THAI_SCRIPT.search(text):
        return ("th", 0.9)
    if FILIPINO_MARKERS.search(text):
        return ("tl", 0.6)

    # FIX: Check both Malay and Indonesian markers and compare
    indo_hits = len(INDONESIAN_MARKERS.findall(text))
    malay_hits = len(MALAY_MARKERS.findall(text))

    if indo_hits > malay_hits:
        return ("id", 0.5 + min(0.3, indo_hits * 0.05))
    if malay_hits > 0:
        return ("ms", 0.5 + min(0.3, malay_hits * 0.05))

    return ("en", 0.4)


def _detect_secondary_languages(text: str, primary: str) -> list[str]:
    """Detect code-switched languages present alongside the primary."""
    secondary = []

    if primary != "en":
        english_hits = len(ENGLISH_MARKERS.findall(text))
        words = text.split()
        if words and english_hits / len(words) > 0.1:
            secondary.append("en")

    if primary == "en":
        if MALAY_MARKERS.search(text):
            secondary.append("ms")
        if INDONESIAN_MARKERS.search(text):
            secondary.append("id")
        if FILIPINO_MARKERS.search(text):
            secondary.append("tl")
        if THAI_SCRIPT.search(text):
            secondary.append("th")

    return secondary


def _detect_dialect_early(text: str) -> str:
    """
    FIX: Check for dialect markers BEFORE langdetect runs.
    This catches Kelantan text that langdetect misclassifies as Indonesian.
    """
    kelantan_matches = len(KELANTAN_MARKERS.findall(text))
    if kelantan_matches >= 2:
        return "kelantan"

    # Kham Mueang (check Thai script presence too)
    if THAI_SCRIPT.search(text):
        kham_matches = len(KHAM_MUEANG_MARKERS.findall(text))
        if kham_matches >= 2:
            return "kham_mueang"

    return "standard"


def _detect_dialect(text: str, primary: str) -> str:
    """Check for regional dialect markers (after language classification)."""
    # Kelantan — accept ms or id (langdetect sometimes misclassifies Kelantan as id)
    if primary in ("ms", "id") and KELANTAN_MARKERS.search(text):
        matches = len(KELANTAN_MARKERS.findall(text))
        if matches >= 1:
            return "kelantan"

    # Javanese — accept id or jv
    if primary in ("id", "jv") and JAVANESE_MARKERS.search(text):
        matches = len(JAVANESE_MARKERS.findall(text))
        if matches >= 1:
            return "javanese"

    # Waray — accept tl or en (langdetect may misclassify)
    if primary in ("tl", "en") and WARAY_MARKERS.search(text):
        matches = len(WARAY_MARKERS.findall(text))
        if matches >= 1:
            return "waray"

    # Kham Mueang (Northern Thai) — accept th
    if primary == "th" and KHAM_MUEANG_MARKERS.search(text):
        matches = len(KHAM_MUEANG_MARKERS.findall(text))
        if matches >= 1:
            return "kham_mueang"

    return "standard"


def _refine_malay_indonesian(text: str, langdetect_guess: str) -> str:
    """
    FIX: Improved Malay vs Indonesian refinement.
    Uses only EXCLUSIVE markers (words unique to each language).
    No longer defaults to Malay — respects langdetect's initial guess on ties.
    """
    malay_score = len(MALAY_MARKERS.findall(text))
    indo_score = len(INDONESIAN_MARKERS.findall(text))

    # FIX: Reduced the margin from +2 to +1 for more sensitive detection
    if indo_score > malay_score + 1:
        return "id"
    elif malay_score > indo_score + 1:
        return "ms"

    # FIX: On tie/close scores, respect langdetect's original guess
    # instead of always defaulting to "ms"
    return langdetect_guess