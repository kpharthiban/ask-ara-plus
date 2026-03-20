"""
AskAra+ — Load Chunks into ChromaDB

Reads chunks.json (output of chunk_documents.py) and loads them into
the ChromaDB `gov_documents` collection.

HOW TO USE (for Lineysha):
──────────────────────────
1. First run:  python data/scripts/chunk_documents.py
2. Then run:   python data/scripts/load_chromadb.py
3. Verify with: python data/scripts/test_retrieval.py

OPTIONS:
  --clear    Wipe the collection before loading (fresh start)
  --file     Path to chunks JSON (default: data/documents/chunks.json)

EXAMPLES:
  python data/scripts/load_chromadb.py                     # Load / upsert
  python data/scripts/load_chromadb.py --clear             # Wipe + reload
  python data/scripts/load_chromadb.py --file my_chunks.json
"""

import json
import os
import sys
import argparse
from pathlib import Path

# Add backend to path so we can import db.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from db import get_collection, add_chunks  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_CHUNKS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "documents", "chunks.json"
)
BATCH_SIZE = 100  # ChromaDB handles batches well; avoids memory spikes


def load(chunks_file: str, clear: bool = False):
    collection = get_collection()

    if clear:
        # Delete all existing documents
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            print(f"[Loader] Cleared {len(existing['ids'])} existing chunks")
        else:
            print("[Loader] Collection already empty")

    # Read chunks
    chunks_path = os.path.abspath(chunks_file)
    if not os.path.exists(chunks_path):
        print(f"[Loader] ERROR: {chunks_path} not found!")
        print("         Run chunk_documents.py first.")
        sys.exit(1)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[Loader] Loading {len(chunks)} chunks from {chunks_path}")

    # Load in batches
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        ids = [c["id"] for c in batch]
        documents = [c["document"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        # Clean metadata: ChromaDB doesn't accept None values
        for meta in metadatas:
            for key, val in list(meta.items()):
                if val is None:
                    meta[key] = ""

        add_chunks(ids, documents, metadatas)
        print(f"  Batch {i // BATCH_SIZE + 1}: loaded {len(batch)} chunks")

    final_count = collection.count()
    print(f"\n[Loader] Done! Collection now has {final_count} chunks total.")
    print(f"[Loader] Next step: python data/scripts/test_retrieval.py")


def main():
    parser = argparse.ArgumentParser(description="Load document chunks into ChromaDB")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the collection before loading",
    )
    parser.add_argument(
        "--file",
        default=DEFAULT_CHUNKS_FILE,
        help="Path to chunks JSON file",
    )
    args = parser.parse_args()
    load(args.file, args.clear)


if __name__ == "__main__":
    main()