"""
assess_complexity — Assess reading complexity of text.

Owner: Tool code by Pharthiban, prompt tuning by Lineysha (#42)
Depends on: glossaries/*.json (for jargon detection)

Approach:
  - Flesch-Kincaid doesn't work for SEA languages (syllable rules are English-specific).
  - Instead we use a composite heuristic:
      1. Average sentence length (words per sentence)
      2. Average word length (characters per word)
      3. Jargon density (glossary terms found ÷ total words)
  - These three signals are weighted into a grade level (1-16 scale).
  - The glossary jargon check scans ALL country glossaries so we catch
    formal terms regardless of which country's document the text came from.
"""

import json
import os
import re
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger("askara.complexity")

# ── Glossary directory ───────────────────────────────────────
# backend/tools/complexity.py → .parent = tools/ → .parent = backend/ → .parent = project root
GLOSSARY_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "glossaries"


# ── Load glossaries (cached) ────────────────────────────────
@lru_cache(maxsize=1)
def _load_all_jargon_terms() -> dict[str, str]:
    """Load formal → simple mappings from ALL country glossaries.

    Returns a dict: { "formal_term_lowered": "simple_equivalent", ... }
    Merges: glossary_my.json, glossary_id.json, glossary_ph.json, glossary_th.json
    """
    terms: dict[str, str] = {}

    if not GLOSSARY_DIR.is_dir():
        logger.warning("Glossary directory not found: %s", GLOSSARY_DIR)
        return terms

    for path in sorted(GLOSSARY_DIR.glob("glossary_*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            count = 0
            # Lineysha's format: {"glossary": {"DOC_NAME": [{"term": ..., "explanation": ...}]}}
            glossary_root = data.get("glossary", data) if isinstance(data, dict) else data

            if isinstance(glossary_root, dict):
                # Nested by document name
                for doc_name, entries in glossary_root.items():
                    if not isinstance(entries, list):
                        continue
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        # Support both formats: term/explanation (Lineysha) and formal/simple (original spec)
                        formal = (entry.get("term") or entry.get("formal", "")).strip()
                        simple = (entry.get("explanation") or entry.get("simple", "")).strip()
                        if formal:
                            terms[formal.lower()] = simple
                            count += 1
            elif isinstance(glossary_root, list):
                # Flat list format (fallback)
                for entry in glossary_root:
                    if not isinstance(entry, dict):
                        continue
                    formal = (entry.get("term") or entry.get("formal", "")).strip()
                    simple = (entry.get("explanation") or entry.get("simple", "")).strip()
                    if formal:
                        terms[formal.lower()] = simple
                        count += 1

            logger.info("Loaded %d terms from %s", count, path.name)
        except Exception as e:
            logger.error("Failed to load glossary %s: %s", path.name, e)

    logger.info("Total jargon terms loaded: %d", len(terms))
    return terms


# ── Sentence splitting (multilingual-aware) ──────────────────
def _split_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Handles common SEA-language patterns:
    - Period, question mark, exclamation mark as terminators
    - Avoids splitting on abbreviations like "No.", "RM.", "Bil."
    """
    # Protect common abbreviations from being split
    protected = text
    abbreviations = [
        "No.", "Bil.", "RM.", "Rp.", "Dr.", "Sdn.", "Bhd.",
        "Inc.", "Ltd.", "Jr.", "Sr.", "vs.", "etc.",
    ]
    for abbr in abbreviations:
        protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))

    # Split on sentence-ending punctuation
    sentences = re.split(r'[.!?।\n]+', protected)

    # Restore dots and clean up
    result = []
    for s in sentences:
        s = s.replace("<DOT>", ".").strip()
        if s and len(s.split()) >= 2:  # Skip fragments
            result.append(s)

    return result if result else [text.strip()]


# ── Core metric calculations ─────────────────────────────────
def _avg_sentence_length(sentences: list[str]) -> float:
    """Average number of words per sentence."""
    if not sentences:
        return 0.0
    total_words = sum(len(s.split()) for s in sentences)
    return total_words / len(sentences)


def _avg_word_length(text: str) -> float:
    """Average character count per word."""
    words = text.split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def _find_jargon(text: str, jargon_map: dict[str, str]) -> list[dict]:
    """Find glossary jargon terms present in the text.

    Checks multi-word terms first (longer matches take priority),
    then single-word terms. Case-insensitive matching.

    Returns list of {"term": ..., "simple": ...} dicts.
    """
    text_lower = text.lower()
    found: list[dict] = []
    matched_spans: list[tuple[int, int]] = []

    # Sort by term length descending so multi-word matches take priority
    sorted_terms = sorted(jargon_map.items(), key=lambda x: len(x[0]), reverse=True)

    for formal_lower, simple in sorted_terms:
        # Skip very short terms (< 3 chars) to avoid false positives
        if len(formal_lower) < 3:
            continue

        start = 0
        while True:
            idx = text_lower.find(formal_lower, start)
            if idx == -1:
                break

            end = idx + len(formal_lower)

            # Check if this span overlaps with an already-matched span
            overlaps = any(
                not (end <= ms or idx >= me)
                for ms, me in matched_spans
            )
            if not overlaps:
                found.append({"term": formal_lower, "simple": simple})
                matched_spans.append((idx, end))
                break  # Only count each term once

            start = end

    return found


def _compute_grade_level(
    avg_sent_len: float,
    avg_word_len: float,
    jargon_density: float,
) -> int:
    """Compute a reading grade level (1-16) from the three metrics.

    Heuristic weights (tuned for SEA government documents):
      - Sentence length is the strongest predictor of difficulty
      - Jargon density indicates domain-specific complexity
      - Word length is a weaker but consistent signal

    Reference points:
      - Grade 3-5 (simple): avg_sent ≈ 8-12, low jargon, short words
      - Grade 8-10 (moderate): avg_sent ≈ 15-20, some jargon
      - Grade 12-16 (complex): avg_sent ≈ 25+, high jargon density
    """
    # Sentence length component (0-8 points)
    if avg_sent_len <= 8:
        sent_score = 1
    elif avg_sent_len <= 12:
        sent_score = 3
    elif avg_sent_len <= 18:
        sent_score = 5
    elif avg_sent_len <= 25:
        sent_score = 7
    else:
        sent_score = 8

    # Word length component (0-4 points)
    if avg_word_len <= 4:
        word_score = 1
    elif avg_word_len <= 6:
        word_score = 2
    elif avg_word_len <= 8:
        word_score = 3
    else:
        word_score = 4

    # Jargon density component (0-4 points)
    if jargon_density <= 0.01:
        jargon_score = 0
    elif jargon_density <= 0.03:
        jargon_score = 1
    elif jargon_density <= 0.06:
        jargon_score = 2
    elif jargon_density <= 0.10:
        jargon_score = 3
    else:
        jargon_score = 4

    grade = sent_score + word_score + jargon_score
    return max(1, min(16, grade))


# ── Main function ────────────────────────────────────────────
def assess_complexity(text: str) -> str:
    """Assess the reading complexity of a text passage.

    Use this tool to check if retrieved document text needs simplification
    before presenting to the user. If the grade level is above 6, the
    agent should call simplify() before showing the text.

    The assessment checks:
    - Average sentence length (words per sentence)
    - Average word length (characters per word)
    - Presence of jargon or technical terms (matched against glossaries)
    - Overall readability grade level (1-16 scale)

    Args:
        text: The text to assess.

    Returns:
        JSON string:
        {
            "grade_level": 12,
            "avg_sentence_length": 24.5,
            "avg_word_length": 5.8,
            "difficult_terms": [
                {"term": "caruman", "simple": "monthly payment"},
                {"term": "skim bencana pekerjaan", "simple": "work accident insurance"}
            ],
            "jargon_count": 3,
            "total_words": 120,
            "jargon_density": 0.025,
            "suggestion": "needs_simplification" | "acceptable" | "already_simple"
        }
    """
    if not text or not text.strip():
        return json.dumps({
            "grade_level": 1,
            "avg_sentence_length": 0,
            "avg_word_length": 0,
            "difficult_terms": [],
            "jargon_count": 0,
            "total_words": 0,
            "jargon_density": 0.0,
            "suggestion": "already_simple",
        })

    # Load jargon terms from all glossaries
    jargon_map = _load_all_jargon_terms()

    # Calculate metrics
    sentences = _split_sentences(text)
    words = text.split()
    total_words = len(words)

    avg_sent_len = round(_avg_sentence_length(sentences), 1)
    avg_word_len = round(_avg_word_length(text), 1)

    # Find jargon terms
    difficult_terms = _find_jargon(text, jargon_map)
    jargon_count = len(difficult_terms)
    jargon_density = round(jargon_count / max(total_words, 1), 4)

    # Compute composite grade level
    grade_level = _compute_grade_level(avg_sent_len, avg_word_len, jargon_density)

    # Decision: does it need simplification?
    if grade_level <= 5:
        suggestion = "already_simple"
    elif grade_level <= 8:
        suggestion = "acceptable"
    else:
        suggestion = "needs_simplification"

    result = {
        "grade_level": grade_level,
        "avg_sentence_length": avg_sent_len,
        "avg_word_length": avg_word_len,
        "difficult_terms": difficult_terms,
        "jargon_count": jargon_count,
        "total_words": total_words,
        "jargon_density": jargon_density,
        "suggestion": suggestion,
    }

    logger.info(
        "assess_complexity: grade=%d, jargon=%d/%d, suggestion=%s",
        grade_level, jargon_count, total_words, suggestion,
    )

    return json.dumps(result, ensure_ascii=False)