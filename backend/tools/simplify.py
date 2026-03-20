"""
simplify_text — Simplify complex official text to a target reading level.

Owner: Lineysha (NLP logic)

Pipeline:
1. Load country glossary
2. Replace formal jargon using glossary
3. Send to LLM with simplification prompt
4. Measure readability before vs after
5. Return structured JSON
"""

import json
import os
import re
import sys
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm_client import call_llm

PROMPT_VERSION = 1
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)

PROMPT_PATH = os.path.join(_BACKEND_DIR, "prompts", f"simplify_v{PROMPT_VERSION}.txt")
GLOSSARY_DIR = os.path.join(_PROJECT_ROOT, "data", "glossaries")


def load_glossary(country: str) -> List[Dict]:
    """Load glossary for a specific country and flatten domains.

    FIX: Handles BOTH glossary formats:
      Format A (glossary_my.json actual):
        { "social_security": [ {"term": "...", "explanation": "..."} ], ... }
      Format B (spec'd but not yet used):
        { "glossary": { "domain": [ {"formal": "...", "simple": "..."} ] } }
    """
    if not country:
        return []

    path = os.path.join(GLOSSARY_DIR, f"glossary_{country.lower()}.json")
    if not os.path.exists(path):
        print(f"[simplify] Glossary not found: {path} — skipping jargon replacement")
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[simplify] Failed to load glossary: {e}")
        return []

    glossary_entries = []

    # FIX: Handle nested {"glossary": {...}} wrapper if present
    if "glossary" in data and isinstance(data["glossary"], dict):
        data = data["glossary"]

    for domain_key, entries in data.items():
        if not isinstance(entries, list):
            continue
        for item in entries:
            if not isinstance(item, dict):
                continue
            # FIX: Handle BOTH field naming conventions
            formal = item.get("formal") or item.get("term", "")
            simple = item.get("simple") or item.get("explanation", "")
            if formal and simple:
                glossary_entries.append({
                    "formal": formal,
                    "simple": simple,
                    "domain": domain_key
                })

    return glossary_entries


def replace_jargon(text: str, glossary: List[Dict]) -> Tuple[str, List[Dict]]:
    processed = text
    replaced_terms = []
    sorted_glossary = sorted(glossary, key=lambda e: len(e["formal"]), reverse=True)
    for entry in sorted_glossary:
        formal = entry["formal"]
        simple = entry["simple"]
        pattern = re.compile(re.escape(formal), re.IGNORECASE)
        if pattern.search(processed):
            processed = pattern.sub(simple, processed)
            replaced_terms.append({
                "original": formal,
                "replacement": simple,
                "domain": entry["domain"]
            })
    return processed, replaced_terms


def avg_words_per_sentence(text: str) -> float:
    sentences = re.split(r"[.!?]", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) == 0:
        return 0.0
    word_count = len(text.split())
    return round(word_count / len(sentences), 2)


def load_prompt() -> str:
    if not os.path.exists(PROMPT_PATH):
        print(f"[simplify] Prompt file not found: {PROMPT_PATH} — using default prompt")
        return (
            "Simplify the following government text to a Grade {level} reading level.\n"
            "Use short sentences. Replace jargon with everyday words.\n"
            "Keep all facts accurate. Output ONLY the simplified text.\n\n"
            "Text:\n{text}"
        )
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


async def simplify_text(
    text: str,
    target_grade_level: int = 5,
    country: str = "",
    language: str = "",
) -> str:
    print("[simplify] Tool executed")
    glossary = load_glossary(country)
    processed_text, replaced_terms = replace_jargon(text, glossary)
    original_complexity = avg_words_per_sentence(text)

    prompt_template = load_prompt()
    prompt = prompt_template.format(
        level=target_grade_level,
        text=processed_text,
        language=language,
        country=country
    )

    system_prompt = (
        "You are a text simplification assistant for AskAra+, "
        "an ASEAN government services helper. "
        "Your job is to rewrite complex official text so that "
        "anyone with a Grade 5 reading level can understand it. "
        "Preserve all factual information. Output ONLY the simplified text. "
        "NEVER include Chinese, Japanese, or Korean characters in your output "
        "unless the target language is one of those."
    )

    try:
        simplified_text = await call_llm(prompt, system_prompt=system_prompt)
    except Exception as e:
        print(f"[simplify] LLM call failed: {e}")
        simplified_text = processed_text

    simplified_text = re.sub(r'\[([^\]]*)\]\(\[link removed\]\)', r'\1', simplified_text)
    simplified_text = re.sub(r'\(\[link removed\]\)', '', simplified_text)
    simplified_text = re.sub(r'\[link removed\]', '', simplified_text)

    cjk_languages = {"zh", "ja", "ko"}
    lang_base = (language or "").split("-")[0].lower()
    if lang_base not in cjk_languages:
        simplified_text = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+', '', simplified_text)

    simplified_text = re.sub(r' {2,}', ' ', simplified_text).strip()
    simplified_complexity = avg_words_per_sentence(simplified_text)

    result = {
        "simplified_text": simplified_text,
        "terms_replaced": replaced_terms,
        "original_grade_level": round(original_complexity),
        "simplified_grade_level": target_grade_level,
        "original_avg_words_per_sentence": original_complexity,
        "simplified_avg_words_per_sentence": simplified_complexity,
        "country": country,
        "language": language,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import asyncio
    test_text = (
        "Pekerja yang layak boleh memohon Skim Bencana Pekerjaan di bawah PERKESO. "
        "Caruman bulanan perlu dibayar oleh majikan. "
        "Permohonan hendaklah dikemukakan dalam tempoh 60 hari dari tarikh kemalangan."
    )
    result = asyncio.run(simplify_text(test_text, country="MY", language="ms"))
    print(result)