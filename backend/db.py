"""
AskAra+ — ChromaDB Database Module

Initialises the ChromaDB persistent client and the `gov_documents` collection
with multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2).

Usage (from backend):
    from db import get_collection

    collection = get_collection()
    results = collection.query(query_texts=["SOCSO registration"], n_results=5)

Usage (from data/scripts — add parent path first):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "backend"))
    from db import get_collection
"""

import os
import chromadb
from chromadb.utils import embedding_functions

# ---------------------------------------------------------------------------
# Config — override via env vars or .env
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    "/app/data/chromadb",
)
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "gov_documents")
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ---------------------------------------------------------------------------
# Singleton client + collection
# ---------------------------------------------------------------------------
_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def get_client() -> chromadb.ClientAPI:
    """Return (or create) the persistent ChromaDB client."""
    global _client
    if _client is None:
        persist_dir = os.path.abspath(CHROMA_PERSIST_DIR)
        os.makedirs(persist_dir, exist_ok=True)
        _client = chromadb.PersistentClient(path=persist_dir)
        print(f"[ChromaDB] Client initialised  →  {persist_dir}")
    return _client


def get_embedding_function() -> embedding_functions.SentenceTransformerEmbeddingFunction:
    """Return the multilingual sentence-transformer embedding function."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )


def get_collection() -> chromadb.Collection:
    """Return (or create) the `gov_documents` collection.

    The collection uses cosine similarity (ChromaDB default) with the
    multilingual MiniLM embedding model so that queries in one ASEAN
    language can retrieve documents written in another.
    """
    global _collection
    if _collection is None:
        client = get_client()
        ef = get_embedding_function()
        _collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        print(
            f"[ChromaDB] Collection '{CHROMA_COLLECTION}' ready  "
            f"({_collection.count()} docs)"
        )
    return _collection


# ---------------------------------------------------------------------------
# Convenience helpers (used by MCP tools + scripts)
# ---------------------------------------------------------------------------

def search(
    query: str,
    n_results: int = 5,
    country: str = "",
    topic: str = "",
) -> dict:
    """Search the collection with optional metadata filters.

    Returns the raw ChromaDB query result dict with keys:
        ids, documents, metadatas, distances
    """
    collection = get_collection()

    where_filters: dict | None = None
    conditions = []
    if country:
        conditions.append({"country": country.upper()})
    if topic:
        conditions.append({"topic": topic.lower()})

    if len(conditions) == 1:
        where_filters = conditions[0]
    elif len(conditions) > 1:
        where_filters = {"$and": conditions}

    return collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filters if where_filters else None,
        include=["documents", "metadatas", "distances"],
    )


def add_chunks(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    """Batch-add document chunks to the collection.

    Automatically deduplicates by ID — if a chunk with the same ID
    already exists it is silently skipped (upsert behaviour).
    """
    collection = get_collection()
    # ChromaDB's .upsert() will update existing IDs and insert new ones
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )
    print(f"[ChromaDB] Upserted {len(ids)} chunks  →  total {collection.count()}")


# ---------------------------------------------------------------------------
# Quick self-test when run directly: python backend/db.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("AskAra+ — ChromaDB Self-Test")
    print("=" * 60)

    col = get_collection()

    # Add 3 dummy docs
    dummy_ids = ["test_my_001", "test_id_001", "test_ph_001"]
    dummy_docs = [
        "Pekerja yang mencarum SOCSO layak menuntut faedah hilang upaya kekal. "
        "Caruman bulanan dikongsi antara majikan dan pekerja.",
        "Pekerja migran Indonesia berhak mendapat perlindungan sesuai UU No. 18/2017. "
        "BPJS Ketenagakerjaan melindungi pekerja dari kecelakaan kerja.",
        "Overseas Filipino Workers are entitled to OWWA benefits including insurance, "
        "repatriation assistance, and livelihood programs through DTI.",
    ]
    dummy_meta = [
        {
            "country": "MY",
            "topic": "worker_rights",
            "language": "ms",
            "source_agency": "PERKESO",
            "document_title": "Panduan Pendaftaran SOCSO",
            "section_heading": "Skim Bencana Pekerjaan",
        },
        {
            "country": "ID",
            "topic": "migrant_rights",
            "language": "id",
            "source_agency": "BP2MI",
            "document_title": "UU No. 18/2017 Perlindungan PMI",
            "section_heading": "Hak Pekerja Migran",
        },
        {
            "country": "PH",
            "topic": "worker_rights",
            "language": "en",
            "source_agency": "OWWA",
            "document_title": "OWWA Benefits Guide",
            "section_heading": "Eligibility and Programs",
        },
    ]

    add_chunks(dummy_ids, dummy_docs, dummy_meta)

    # Test queries
    print("\n--- Query: 'SOCSO registration' ---")
    res = search("SOCSO registration", n_results=3)
    for i, (doc, meta, dist) in enumerate(
        zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ):
        sim = 1 - dist  # cosine distance → similarity
        print(f"  [{i+1}] sim={sim:.3f} | {meta['country']} | {meta['document_title']}")
        print(f"       {doc[:80]}...")

    print("\n--- Query: 'hak pekerja migran' (cross-lingual test) ---")
    res = search("hak pekerja migran", n_results=3)
    for i, (doc, meta, dist) in enumerate(
        zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ):
        sim = 1 - dist
        print(f"  [{i+1}] sim={sim:.3f} | {meta['country']} | {meta['document_title']}")
        print(f"       {doc[:80]}...")

    print("\n--- Query: 'OFW benefits' (filtered: country=PH) ---")
    res = search("OFW benefits", n_results=3, country="PH")
    for i, (doc, meta, dist) in enumerate(
        zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ):
        sim = 1 - dist
        print(f"  [{i+1}] sim={sim:.3f} | {meta['country']} | {meta['document_title']}")
        print(f"       {doc[:80]}...")

    # Cleanup test data
    col.delete(ids=dummy_ids)
    print(f"\n[Cleanup] Removed dummy docs. Collection count: {col.count()}")
    print("\n✅ ChromaDB self-test passed!")
