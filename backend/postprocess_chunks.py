#!/usr/bin/env python3
"""
postprocess_chunks.py — AskAra+ chunks.json quality filter
===========================================================
Runs AFTER chunk_documents.py, BEFORE load_chromadb.py.

Fixes three problems found in the audit:
  1. Garbage chunks (< MIN_CHUNK_WORDS words)          — removes 191–496 bad chunks
  2. Duplicate chunk content (same block in many docs) — removes 109+ dupes
  3. Big-Act dominance (EPF/Akta Kerja/OSH = 44.6%)   — caps chunks per source doc

Usage:
    python postprocess_chunks.py                        # reads + writes chunks.json in-place
    python postprocess_chunks.py --src path/to/chunks.json --out path/to/chunks_clean.json
    python postprocess_chunks.py --dry-run              # stats only, no write
    python postprocess_chunks.py --no-cap               # skip per-doc cap (keep all good chunks)
"""

import argparse
import hashlib
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────
MIN_CHUNK_WORDS   = 30    # drop chunks with fewer words than this
MAX_CHUNKS_PER_DOC = 80   # cap per source document (prevents EPF Act dominating)
DEDUP_CHARS       = 150   # fingerprint length for duplicate detection
# ──────────────────────────────────────────────────────────────────────────────

# Section headings that are pure boilerplate — chunks with ONLY these headings
# and minimal body text should be dropped regardless of word count.
BOILERPLATE_HEADINGS = {
    "DOKUMEN DAN PAUTAN BERKAITAN",
    "DOKUMEN PDF BERKAITAN",
    "DOKUMEN BERKAITAN",
    "FOR THE MONTH",
    "BORANG RUJUKAN",
    "Introduction",       # catches the ASEAN doc "Introduction\ni i" garbage chunk
    "JADUAL",
}


def chunk_fingerprint(text: str) -> str:
    """MD5 of first DEDUP_CHARS chars, lowercased and whitespace-normalised."""
    normalised = " ".join(text.lower().split())[:DEDUP_CHARS]
    return hashlib.md5(normalised.encode()).hexdigest()


def is_boilerplate_chunk(chunk: dict) -> bool:
    """True if the chunk is a known sidebar/boilerplate section with no real content."""
    heading = chunk.get("metadata", {}).get("section_heading", "")
    if heading in BOILERPLATE_HEADINGS:
        # Allow through if body has substantial content beyond the heading line
        body_words = len(chunk["document"].split())
        if body_words < MIN_CHUNK_WORDS:
            return True
    return False


def postprocess(
    chunks: list[dict],
    min_words: int = MIN_CHUNK_WORDS,
    max_per_doc: int = MAX_CHUNKS_PER_DOC,
    apply_cap: bool = True,
) -> tuple[list[dict], dict]:
    """
    Returns (filtered_chunks, stats_dict).
    """
    stats = {
        "input": len(chunks),
        "removed_too_short": 0,
        "removed_boilerplate_heading": 0,
        "removed_duplicate": 0,
        "removed_doc_cap": 0,
    }

    # ── Pass 1: word-count filter + boilerplate heading filter ────────────────
    pass1 = []
    for c in chunks:
        word_count = len(c["document"].split())
        if word_count < min_words:
            stats["removed_too_short"] += 1
            continue
        if is_boilerplate_chunk(c):
            stats["removed_boilerplate_heading"] += 1
            continue
        pass1.append(c)

    # ── Pass 2: content deduplication (across all docs) ──────────────────────
    seen_fps: set[str] = set()
    pass2 = []
    for c in pass1:
        fp = chunk_fingerprint(c["document"])
        if fp in seen_fps:
            stats["removed_duplicate"] += 1
            continue
        seen_fps.add(fp)
        pass2.append(c)

    # ── Pass 3: per-document cap ──────────────────────────────────────────────
    if apply_cap:
        doc_counts: Counter = Counter()
        pass3 = []
        for c in pass2:
            doc_title = c["metadata"].get("document_title", "")
            if doc_counts[doc_title] < max_per_doc:
                doc_counts[doc_title] += 1
                pass3.append(c)
            else:
                stats["removed_doc_cap"] += 1
    else:
        pass3 = pass2

    stats["output"] = len(pass3)
    stats["removed_total"] = stats["input"] - stats["output"]
    return pass3, stats


def main():
    parser = argparse.ArgumentParser(description="Post-process chunks.json quality")
    parser.add_argument("--src",      default="../data/documents/chunks.json",      help="Input chunks.json")
    parser.add_argument("--out",      default=None,                        help="Output path (default: overwrite src)")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--no-cap",   action="store_true",                 help="Skip per-doc cap")
    parser.add_argument("--min-words", type=int, default=MIN_CHUNK_WORDS)
    parser.add_argument("--max-per-doc", type=int, default=MAX_CHUNKS_PER_DOC)
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out) if args.out else src

    if not src.exists():
        log.error(f"Not found: {src}")
        return

    log.info(f"Reading {src} ...")
    with open(src, encoding="utf-8") as f:
        chunks = json.load(f)

    filtered, stats = postprocess(
        chunks,
        min_words=args.min_words,
        max_per_doc=args.max_per_doc,
        apply_cap=not args.no_cap,
    )

    # ── Report ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  CHUNKS POSTPROCESS REPORT" + ("  (DRY RUN)" if args.dry_run else ""))
    print("="*60)
    print(f"  Input chunks  : {stats['input']:>5}")
    print(f"  Output chunks : {stats['output']:>5}  ({100*stats['output']/stats['input']:.1f}% kept)")
    print(f"  ─────────────────────────────────────")
    print(f"  Removed < {args.min_words}w  : {stats['removed_too_short']:>5}")
    print(f"  Removed boilerplate : {stats['removed_boilerplate_heading']:>5}")
    print(f"  Removed duplicates  : {stats['removed_duplicate']:>5}")
    print(f"  Removed (doc cap)   : {stats['removed_doc_cap']:>5}")
    print(f"  Total removed       : {stats['removed_total']:>5}")

    # Per-doc breakdown for capped docs
    if not args.no_cap:
        doc_counts_in  = Counter(c["metadata"].get("document_title") for c in chunks)
        doc_counts_out = Counter(c["metadata"].get("document_title") for c in filtered)
        capped = {d: (doc_counts_in[d], doc_counts_out[d])
                  for d in doc_counts_in
                  if doc_counts_in[d] > args.max_per_doc}
        if capped:
            print(f"\n  Docs that hit the {args.max_per_doc}-chunk cap:")
            for doc, (before, after) in sorted(capped.items(), key=lambda x: -x[1][0]):
                print(f"    {before:>3} → {after:>3}  {doc[:60]}")

    # Word-count distribution of output
    if filtered:
        wcs = [len(c["document"].split()) for c in filtered]
        print(f"\n  Output chunk word distribution:")
        print(f"    < 30w  : {sum(1 for w in wcs if w < 30):>4}  (should be 0)")
        print(f"    30–99w : {sum(1 for w in wcs if 30 <= w < 100):>4}")
        print(f"    100w+  : {sum(1 for w in wcs if w >= 100):>4}")
        print(f"    avg    : {sum(wcs)/len(wcs):.0f}w")

    print("="*60)

    if args.dry_run:
        print("  DRY RUN — no file written.\n")
        return

    with open(out, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    log.info(f"Written {len(filtered)} chunks → {out}")
    print(f"\n  Next step: uv run python load_chromadb.py --clear\n")


if __name__ == "__main__":
    main()