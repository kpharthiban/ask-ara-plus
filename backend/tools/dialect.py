"""
dialect_adapt — Adapt standard language text to a regional dialect.

Owner: Pharthiban (tool code), Lineysha (dialect JSONs + prompt tuning)
Depends on: llm_client.py, data/glossaries/dialect_*.json
"""

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger("askara.dialect")

# FIX: Try multiple paths to find glossary files
_THIS_FILE = Path(__file__).resolve()
_POSSIBLE_ROOTS = [
    _THIS_FILE.parent.parent.parent,   # tools/dialect.py -> backend -> project root
    _THIS_FILE.parent.parent,           # backend/dialect.py -> project root
    _THIS_FILE.parent,                  # flat layout
]

GLOSSARY_DIR = None
for root in _POSSIBLE_ROOTS:
    candidate = root / "data" / "glossaries"
    if candidate.is_dir():
        GLOSSARY_DIR = candidate
        break

if GLOSSARY_DIR is None:
    GLOSSARY_DIR = _THIS_FILE.parent.parent.parent / "data" / "glossaries"
    logger.warning("Glossary directory not found, using: %s", GLOSSARY_DIR)


DIALECT_INFO = {
    "kelantan_malay": {
        "base_lang": "ms",
        "name": "Kelantan Malay (Kelate)",
        "region": "Kelantan, Malaysia",
        "glossary_file": "dialect_kelantan.json",
        "few_shot_examples": [
            ("Saya tidak tahu bagaimana nak pergi ke pejabat itu.",
             "Ambo dok tahu macam mano nok gi ke pejabat tu."),
            ("Kamu perlu bawa kad pengenalan dan surat doktor.",
             "Demo keno bawok kad pengenalan nga surat doktor."),
            ("Dia sudah pergi ke rumah sakit semalam.",
             "Dio doh gi ke rumoh sakit semale."),
        ],
    },
    "javanese": {
        "base_lang": "id",
        "name": "Javanese (Basa Jawa)",
        "region": "Central & East Java, Indonesia",
        "glossary_file": "dialect_javanese.json",
        "few_shot_examples": [
            ("Saya tidak tahu bagaimana cara mendaftar.",
             "Aku ora ngerti piye carane ndaftar."),
            ("Kamu harus pergi ke kantor pemerintah.",
             "Kowe kudu lunga nang kantor pemerintah."),
            ("Mereka sudah mendapatkan bantuan dari pemerintah.",
             "Wong-wong wis oleh bantuan saka pemerintah."),
        ],
    },
    "cebuano": {
        "base_lang": "tl",
        "name": "Cebuano (Bisaya)",
        "region": "Cebu & Visayas, Philippines",
        "glossary_file": None,
        "few_shot_examples": [
            ("Pumunta ka sa pinakamalapit na tanggapan.",
             "Adto ka sa pinakaduol nga opisina."),
            ("Dalhin mo ang iyong ID at mga dokumento.",
             "Dad-a ang imong ID ug mga dokumento."),
        ],
    },
    "waray": {
        "base_lang": "tl",
        "name": "Waray (Winaray)",
        "region": "Eastern Visayas, Philippines",
        "glossary_file": "dialect_waray.json",
        "few_shot_examples": [
            ("Pumunta ka sa pinakamalapit na tanggapan.",
             "Kadto ka ha pinakaharani nga opisina."),
            ("Dalhin mo ang iyong ID at mga dokumento.",
             "Dad-a an imo ID ngan mga papeles."),
        ],
    },
    "kham_mueang": {
        "base_lang": "th",
        "name": "Kham Mueang (Northern Thai / Lanna)",
        "region": "Chiang Mai, Chiang Rai, Northern Thailand",
        "glossary_file": "dialect_kham_mueang.json",
        "few_shot_examples": [
            ("\u0e44\u0e1b\u0e17\u0e35\u0e48\u0e2a\u0e33\u0e19\u0e31\u0e01\u0e07\u0e32\u0e19\u0e17\u0e35\u0e48\u0e43\u0e01\u0e25\u0e49\u0e17\u0e35\u0e48\u0e2a\u0e38\u0e14",
             "\u0e44\u0e1b\u0e15\u0e35\u0e49\u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e4b\u0e32\u0e19\u0e17\u0e35\u0e48\u0e43\u0e01\u0e49\u0e17\u0e35\u0e48\u0e2a\u0e38\u0e14"),
        ],
    },
}


@lru_cache(maxsize=4)
def _load_dialect_mapping(dialect_key: str) -> list[tuple[str, str]]:
    """Load vocabulary mapping. FIX: Handles both with and without 'glossary' wrapper."""
    info = DIALECT_INFO.get(dialect_key)
    if not info or not info.get("glossary_file"):
        return []

    glossary_path = GLOSSARY_DIR / info["glossary_file"]
    if not glossary_path.is_file():
        logger.warning("Dialect glossary not found: %s", glossary_path)
        return []

    try:
        with open(glossary_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error("Failed to load dialect glossary %s: %s", glossary_path, e)
        return []

    mappings: list[tuple[str, str]] = []

    # FIX: Handle both with and without "glossary" wrapper
    glossary_root = data
    if isinstance(data, dict) and "glossary" in data:
        glossary_root = data["glossary"]

    if isinstance(glossary_root, dict):
        for _category, entries in glossary_root.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                standard, dialect = _extract_pair(entry, dialect_key)
                if standard and dialect:
                    mappings.append((standard, dialect))
    elif isinstance(glossary_root, list):
        for entry in glossary_root:
            if not isinstance(entry, dict):
                continue
            standard, dialect = _extract_pair(entry, dialect_key)
            if standard and dialect:
                mappings.append((standard, dialect))

    mappings.sort(key=lambda x: len(x[0]), reverse=True)
    logger.info("Loaded %d dialect mappings for %s", len(mappings), dialect_key)
    return mappings


def _extract_pair(entry: dict, dialect_key: str) -> tuple[str, str]:
    standard = ""
    dialect = ""
    if dialect_key == "kelantan_malay":
        standard = entry.get("standard_bm", "")
        dialect = entry.get("kelantan", "")
    elif dialect_key == "javanese":
        standard = entry.get("standard_id", "") or entry.get("standard_indo", "")
        dialect = entry.get("javanese_ngoko", "") or entry.get("jawa", "")
    elif dialect_key == "waray":
        standard = entry.get("standard_tagalog", "") or entry.get("standard_tl", "")
        dialect = entry.get("waray", "")
    elif dialect_key == "kham_mueang":
        standard = entry.get("standard_thai", "") or entry.get("standard_th", "")
        dialect = entry.get("kham_mueang", "")
    if not standard:
        standard = entry.get("standard", "") or entry.get("formal", "") or entry.get("term", "")
    if not dialect:
        dialect = entry.get("dialect", "") or entry.get("informal", "") or entry.get("adaptation", "")
    return standard.strip(), dialect.strip()


def _apply_vocab_mapping(text: str, mappings: list[tuple[str, str]]) -> tuple[str, int]:
    adapted = text
    count = 0
    for standard, dialect in mappings:
        pattern = re.compile(re.escape(standard), re.IGNORECASE)
        matches = pattern.findall(adapted)
        if matches:
            adapted = pattern.sub(dialect, adapted)
            count += len(matches)
    return adapted, count


DIALECT_SYSTEM_PROMPT = """\
You are an expert in Southeast Asian regional dialects and informal speech.
Your job is to rewrite text in a specific regional dialect so it sounds natural
to local speakers.

Rules:
1. Rewrite the text so it sounds natural in the target dialect.
2. Adjust sentence structure, particles, and word endings to match the dialect.
3. Keep the MEANING exactly the same.
4. Keep it SIMPLE — this text is for people with low literacy.
5. Preserve any names, numbers, dates, and reference numbers exactly.
6. Output ONLY the rewritten text — no explanations, no notes.\
"""


def _build_dialect_prompt(text: str, dialect_key: str) -> str:
    info = DIALECT_INFO.get(dialect_key, {})
    dialect_name = info.get("name", dialect_key)
    examples = info.get("few_shot_examples", [])
    prompt_parts = [f"Rewrite the following text in {dialect_name} dialect.\n"]
    if examples:
        prompt_parts.append("Here are examples of how this dialect sounds:\n")
        for standard, dialect in examples:
            prompt_parts.append(f"Standard: {standard}")
            prompt_parts.append(f"Dialect:  {dialect}\n")
    prompt_parts.append(f"Now rewrite this text in {dialect_name}:")
    prompt_parts.append(f"\n{text}")
    return "\n".join(prompt_parts)


async def dialect_adapt(text: str, target_dialect: str) -> str:
    if not text or not text.strip():
        return json.dumps({"adapted_text": "", "target_dialect": target_dialect, "base_lang": "", "words_adapted": 0, "method": "none", "note": "Empty input"})

    if target_dialect not in DIALECT_INFO:
        return json.dumps({"adapted_text": text, "target_dialect": target_dialect, "base_lang": "", "words_adapted": 0, "method": "none",
            "note": f"Unsupported dialect: {target_dialect}. Supported: {', '.join(DIALECT_INFO.keys())}"})

    dialect_info = DIALECT_INFO[target_dialect]
    base_lang = dialect_info["base_lang"]
    dialect_name = dialect_info["name"]

    mappings = _load_dialect_mapping(target_dialect)
    if mappings:
        vocab_adapted, words_adapted = _apply_vocab_mapping(text, mappings)
        has_vocab = True
    else:
        vocab_adapted = text
        words_adapted = 0
        has_vocab = False

    from llm_client import call_llm, LLMError
    prompt = _build_dialect_prompt(vocab_adapted, target_dialect)

    try:
        llm_adapted = await call_llm(prompt, system_prompt=DIALECT_SYSTEM_PROMPT, temperature=0.4, max_tokens=2048)
        llm_adapted = _clean_dialect_output(llm_adapted)
        method = "vocab_mapping+llm" if has_vocab else "llm_only"
        final_text = llm_adapted
    except LLMError as e:
        logger.error("Dialect LLM rewrite failed: %s", e)
        if has_vocab:
            method = "vocab_only"
            final_text = vocab_adapted
        else:
            method = "fallback_passthrough"
            final_text = text

    logger.info("dialect_adapt: %s, method=%s, words_adapted=%d", target_dialect, method, words_adapted)

    return json.dumps({
        "adapted_text": final_text, "target_dialect": target_dialect, "base_lang": base_lang,
        "words_adapted": words_adapted, "method": method,
        "note": f"Adapted to {dialect_name} with {method.replace('_', ' ')}",
    }, ensure_ascii=False)


def _clean_dialect_output(text: str) -> str:
    cleaned = text.strip()
    prefixes = ["Here is the rewritten text:", "Dialect version:", "In Kelantan dialect:", "Versi Kelantan:", "Dalam dialek Kelantan:"]
    for prefix in prefixes:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1].strip()
    return cleaned