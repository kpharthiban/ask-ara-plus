"""
AskAra+ — ChromaDB Ingestion Script
=====================================
Reads plain-text document chunks and their .meta.json sidecars from a data
directory, then upserts them into the ChromaDB `gov_documents` collection.

Malaysia-only policy
---------------------
Only documents with country="MY" or country="ASEAN" are ingested.
ID, PH, and TH documents are skipped and logged — their source files are
kept on disk so they can be re-added if the scope ever expands.

Document format (Lineysha's standard)
---------------------------------------
Each document is a pair of files:
    <Name>.txt            — plain text, ALL-CAPS section headings
    <Name>.meta.json      — sidecar metadata with fields:
                              country, topic, language, source_agency,
                              document_title, effective_date, document_url

Chunking strategy
------------------
Documents are split on ALL-CAPS section headings (blank-line separated).
Each section becomes one chunk. Sections longer than MAX_CHUNK_CHARS are
further split at paragraph boundaries to stay within embedding limits.

Usage
------
    # From backend/ directory:
    uv run python load_chromadb.py

    # With a custom data directory:
    uv run python load_chromadb.py --data-dir ../data/documents

    # Dry run (list files that would be ingested without touching ChromaDB):
    uv run python load_chromadb.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── Bootstrap path so db.py is importable ────────────────────────────────────
# Add backend to path
backend_path = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(backend_path))

# Load .env BEFORE importing db (critical!)
from dotenv import load_dotenv
load_dotenv(backend_path / "../.env")

from db import add_chunks, get_collection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("askara.ingest")

# ── Constants ─────────────────────────────────────────────────────────────────

# Malaysia-only: only ingest these country codes
INGEST_COUNTRIES: set[str] = {"MY", "ASEAN"}

# Characters per chunk — keeps chunks within MiniLM's 512-token window
MAX_CHUNK_CHARS = 1_200

# Minimum chunk length — skip tiny slivers (headings with no body)
MIN_CHUNK_CHARS = 40

# Regex that matches ALL-CAPS section headings (Lineysha's document standard)
# Example: "ELIGIBILITY CRITERIA" or "HOW TO APPLY"
_HEADING_RE = re.compile(r"^[A-Z][A-Z\s/()&,:\-]{4,}$", re.MULTILINE)


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_document(text: str, doc_title: str) -> list[dict]:
    """
    Split a document into chunks by ALL-CAPS section headings.

    Returns a list of dicts:
        {"heading": str, "text": str}
    """
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Split on blank lines to get paragraphs
    paragraphs = re.split(r"\n{2,}", text)

    chunks: list[dict] = []
    current_heading = doc_title  # default heading = document title
    current_body: list[str] = []

    def _flush():
        body = "\n\n".join(current_body).strip()
        if len(body) >= MIN_CHUNK_CHARS:
            chunks.append({"heading": current_heading, "text": body})
        current_body.clear()

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Detect ALL-CAPS heading
        if _HEADING_RE.match(para) and len(para) < 120:
            _flush()
            current_heading = para
        else:
            current_body.append(para)

    _flush()  # flush the last section

    # Split oversized chunks at paragraph boundaries
    final: list[dict] = []
    for chunk in chunks:
        if len(chunk["text"]) <= MAX_CHUNK_CHARS:
            final.append(chunk)
        else:
            final.extend(_split_large_chunk(chunk))

    return final


def _split_large_chunk(chunk: dict) -> list[dict]:
    """Further split a chunk that exceeds MAX_CHUNK_CHARS."""
    parts = []
    paras = chunk["text"].split("\n\n")
    current: list[str] = []
    idx = 1

    for para in paras:
        probe = "\n\n".join(current + [para])
        if len(probe) > MAX_CHUNK_CHARS and current:
            parts.append({
                "heading": f"{chunk['heading']} (part {idx})",
                "text": "\n\n".join(current),
            })
            current = [para]
            idx += 1
        else:
            current.append(para)

    if current:
        parts.append({
            "heading": f"{chunk['heading']} (part {idx})" if idx > 1 else chunk["heading"],
            "text": "\n\n".join(current),
        })

    return parts


# ── File discovery ────────────────────────────────────────────────────────────

def discover_documents(data_dir: Path) -> list[tuple[Path, dict]]:
    """
    Walk data_dir and return (txt_path, metadata) pairs for MY/ASEAN docs only.

    Skips files if:
    - No matching .meta.json sidecar found
    - country not in INGEST_COUNTRIES
    - .meta.json is malformed
    """
    pairs: list[tuple[Path, dict]] = []
    skipped_country: list[str] = []
    skipped_no_meta: list[str] = []
    skipped_bad_meta: list[str] = []

    txt_files = sorted(data_dir.glob("**/*.txt"))

    for txt_path in txt_files:
        # Skip files that are themselves meta files accidentally named .txt
        if txt_path.stem.endswith(".meta"):
            continue

        # Locate sidecar — same stem + "_meta.json"
        meta_path = txt_path.with_name(txt_path.stem + ".meta.json")
        if not meta_path.exists():
            skipped_no_meta.append(txt_path.name)
            continue

        # Parse metadata
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            skipped_bad_meta.append(f"{meta_path.name}: {e}")
            continue

        # Country filter — MY-only policy
        country = meta.get("country", "").strip().upper()
        if country not in INGEST_COUNTRIES:
            skipped_country.append(f"{txt_path.name} (country={country})")
            continue

        pairs.append((txt_path, meta))

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("Document discovery complete:")
    logger.info("  ✅ Will ingest : %d documents", len(pairs))
    logger.info("  ⏭️  Skipped (wrong country) : %d", len(skipped_country))
    logger.info("  ⚠️  Skipped (no meta sidecar): %d", len(skipped_no_meta))
    logger.info("  ❌ Skipped (bad meta JSON)   : %d", len(skipped_bad_meta))

    if skipped_country:
        logger.debug("Non-MY/ASEAN files skipped:\n  %s",
                     "\n  ".join(skipped_country))
    if skipped_no_meta:
        logger.warning("Files without .meta.json sidecar:\n  %s",
                       "\n  ".join(skipped_no_meta))
    if skipped_bad_meta:
        logger.error("Malformed .meta.json files:\n  %s",
                     "\n  ".join(skipped_bad_meta))

    return pairs


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_documents(
    pairs: list[tuple[Path, dict]],
    dry_run: bool = False,
) -> None:
    """
    Chunk and upsert all discovered documents into ChromaDB.

    Chunk ID format:  <country>_<doc_stem>_chunk<N>
    Example:          MY_Akta_Kerja_1955_chunk001
    """
    total_chunks = 0
    total_docs = 0
    errors: list[str] = []

    for txt_path, meta in pairs:
        doc_stem = txt_path.stem
        country  = meta.get("country", "MY").upper()
        doc_title = meta.get("document_title", doc_stem)

        # Read document text
        try:
            text = txt_path.read_text(encoding="utf-8")
        except OSError as e:
            errors.append(f"{txt_path.name}: {e}")
            continue

        # Chunk
        chunks = chunk_document(text, doc_title)
        if not chunks:
            logger.warning("No chunks produced for %s — skipping", txt_path.name)
            continue

        # Build ChromaDB batch
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for i, chunk in enumerate(chunks, start=1):
            chunk_id = f"{country}_{doc_stem}_chunk{i:03d}"
            chunk_text = f"{chunk['heading']}\n\n{chunk['text']}"

            ids.append(chunk_id)
            documents.append(chunk_text)
            metadatas.append({
                # Required metadata fields
                "country":        country,
                "topic":          meta.get("topic", ""),
                "language":       meta.get("language", "ms"),
                "source_agency":  meta.get("source_agency", ""),
                "document_title": doc_title,
                "section_heading": chunk["heading"],
                # Optional metadata fields
                "effective_date": meta.get("effective_date", ""),
                "document_url":   meta.get("document_url", ""),
            })

        if dry_run:
            logger.info(
                "[DRY RUN] %s → %d chunks (country=%s, topic=%s)",
                txt_path.name, len(chunks), country, meta.get("topic", ""),
            )
        else:
            try:
                add_chunks(ids=ids, documents=documents, metadatas=metadatas)
                logger.info(
                    "✅ Ingested %s → %d chunks (country=%s, topic=%s)",
                    txt_path.name, len(chunks), country, meta.get("topic", ""),
                )
            except Exception as e:
                errors.append(f"{txt_path.name}: {e}")
                logger.error("❌ Failed to ingest %s: %s", txt_path.name, e)
                continue

        total_chunks += len(chunks)
        total_docs   += 1

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("AskAra+ — Ingestion Complete")
    print("=" * 60)
    print(f"  Documents processed : {total_docs}")
    print(f"  Total chunks        : {total_chunks}")
    print(f"  Errors              : {len(errors)}")

    if not dry_run:
        col = get_collection()
        print(f"  ChromaDB total docs : {col.count()}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  ❌ {e}")
    else:
        print("\n✅ All documents ingested successfully!")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AskAra+ ChromaDB ingestion — Malaysia-only knowledge base.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "documents",
        help="Directory containing .txt + .meta.json document pairs "
             "(default: ../data/documents)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be ingested without touching ChromaDB.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all existing documents from the collection before ingesting. "
             "USE WITH CAUTION.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    data_dir: Path = args.data_dir.resolve()
    logger.info("Data directory : %s", data_dir)
    logger.info("Ingest countries: %s", ", ".join(sorted(INGEST_COUNTRIES)))
    logger.info("Dry run         : %s", args.dry_run)

    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    # Optional: wipe existing collection
    if args.clear and not args.dry_run:
        col = get_collection()
        existing = col.count()
        if existing > 0:
            logger.warning("--clear specified: deleting %d existing docs...", existing)
            col.delete(where={"country": {"$in": list(INGEST_COUNTRIES)}})
            logger.info("Collection cleared.")

    # Discover → ingest
    pairs = discover_documents(data_dir)

    if not pairs:
        logger.warning("No documents found to ingest. Check your data directory.")
        sys.exit(0)

    ingest_documents(pairs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()