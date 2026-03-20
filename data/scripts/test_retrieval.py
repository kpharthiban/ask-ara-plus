"""
AskAra+ — ChromaDB Retrieval Sanity Check

Quick tests to verify documents are loaded and retrievable.
Run this after load_chromadb.py to confirm everything works.

HOW TO USE (for Lineysha):
──────────────────────────
  python data/scripts/test_retrieval.py

It will:
  1. Show collection stats (total chunks, countries, topics)
  2. Run sample queries in multiple languages
  3. Test metadata filtering
  4. Test cross-lingual retrieval (query in one language → docs in another)
  5. Show similarity scores so you can gauge quality

If results look wrong, check:
  - Did chunk_documents.py produce good chunks? (check chunks.json)
  - Are the .meta.json files correct? (country, topic, language)
  - Try re-running: load_chromadb.py --clear
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from db import get_collection, search  # noqa: E402

SIMILARITY_THRESHOLD = 0.65  # From plan: cosine sim >= 0.65 = confident match


def print_results(label: str, results: dict):
    """Pretty-print query results."""
    print(f"\n{'─' * 60}")
    print(f"  Query: {label}")
    print(f"{'─' * 60}")

    if not results["documents"][0]:
        print("  ❌ No results found!")
        return

    for i, (doc, meta, dist) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ):
        sim = 1 - dist  # cosine distance → similarity
        status = "✅" if sim >= SIMILARITY_THRESHOLD else "⚠️"
        print(
            f"  {status} [{i+1}] sim={sim:.3f} | "
            f"{meta.get('country', '??')} | "
            f"{meta.get('language', '??')} | "
            f"{meta.get('document_title', 'untitled')}"
        )
        print(f"        Section: {meta.get('section_heading', 'N/A')}")
        # Show first 120 chars of the chunk
        preview = doc[:120].replace("\n", " ")
        print(f"        {preview}...")
    print()


def main():
    collection = get_collection()
    total = collection.count()

    print("=" * 60)
    print("  AskAra+ — Retrieval Sanity Check")
    print("=" * 60)
    print(f"\n  Collection: {collection.name}")
    print(f"  Total chunks: {total}")

    if total == 0:
        print("\n  ⚠ Collection is empty! Load documents first:")
        print("    1. python data/scripts/chunk_documents.py")
        print("    2. python data/scripts/load_chromadb.py")
        return

    # --- Stats ---
    all_docs = collection.get(include=["metadatas"])
    countries = set()
    topics = set()
    languages = set()
    for meta in all_docs["metadatas"]:
        countries.add(meta.get("country", "unknown"))
        topics.add(meta.get("topic", "unknown"))
        languages.add(meta.get("language", "unknown"))

    print(f"  Countries: {', '.join(sorted(countries))}")
    print(f"  Topics: {', '.join(sorted(topics))}")
    print(f"  Languages: {', '.join(sorted(languages))}")

    # --- Test Queries ---

    # --- Updated Test Queries based on Uploaded Docs ---

    # 1. Malaysia: SOCSO/PERKESO
    print_results(
        "PERKESO/ASSIST Portal (MY docs expected)",
        search("ASSIST Portal Quick Reference Guides PERKESO", n_results=3),
    )

    # 2. Indonesia: 2026 Social Aid
    print_results(
        "BLT Kemensos 2026 (ID query)",
        search("Bantuan Langsung Tunai Kemensos 2026", n_results=3),
    )

    # 3. Philippines: Crisis Assistance
    print_results(
        "DSWD AICS Crisis Situation (PH query)",
        search("Assistance to Individuals in Crisis Situation AICS", n_results=3),
    )

    # 4. Thailand: Section 40 Benefits
    print_results(
        "SSO Section 40 Old-Age Benefits (TH query)",
        search("Benefits for Insured Persons under Section 40", n_results=3),
    )

    # 5. Cross-lingual / Regional: Migrant Rights
    print_results(
        "Migrant Workers Rights (ASEAN Consensus → Cross-lingual test)",
        search("Protection and Promotion of the Rights of Migrant Workers", n_results=3),
    )   

    # 6. Filtered query: only Malaysian docs
    print_results(
        "worker rights (filtered: country=MY)",
        search("worker rights", n_results=3, country="MY"),
    )

    # 7. Filtered query: only Indonesian docs
    print_results(
        "worker protection (filtered: country=ID)",
        search("worker protection", n_results=3, country="ID"),
    )

    # --- Summary ---
    print("=" * 60)
    print("  DONE — Check the results above.")
    print(f"  Similarity threshold: {SIMILARITY_THRESHOLD}")
    print("  ✅ = above threshold (confident match)")
    print("  ⚠️  = below threshold (weak match)")
    print()
    print("  If most results show ✅, your ChromaDB is working well!")
    print("  If you see lots of ⚠️, check your chunks + metadata.")
    print("=" * 60)


if __name__ == "__main__":
    main()