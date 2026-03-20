"""
search_documents() — Real implementation
Queries ChromaDB via db.py singleton, applies metadata filters,
returns top chunks with source citations.
"""

import json
import logging
from typing import Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# Import your ChromaDB singleton — adjust path if needed
from db import get_collection

# Similarity threshold: below this, results are "not confident"
SIMILARITY_THRESHOLD = 0.40
DEFAULT_N_RESULTS = 5


def search_documents(query: str, country: str = "", topic: str = "") -> str:
    """
    Search the knowledge base for relevant government documents.

    Args:
        query: User's search query (any language — embeddings are multilingual)
        country: ISO filter — "MY", "ID", "PH", "TH", "ASEAN", or "" for all
        topic: Topic filter — e.g. "social_security", "migrant_rights", or "" for all

    Returns:
        JSON string with search results or a no-results message.
    """
    try:
        collection = get_collection()

        # --- Build metadata filter ---
        where_filter = _build_where_filter(country, topic)

        # --- Query ChromaDB ---
        query_kwargs = {
            "query_texts": [query],
            "n_results": DEFAULT_N_RESULTS,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = collection.query(**query_kwargs)

        # --- Process results ---
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        # DEBUG: Log what ChromaDB actually returned
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            sim = 1 - (dist / 2)
            logger.info(
                "search_documents result[%d]: sim=%.4f | %s | %s",
                i, sim,
                meta.get("document_title", "?"),
                doc[:80],
            )

        if not documents:
            return json.dumps({
                "status": "no_results",
                "message": f"No documents found for query: '{query}'"
                           + (f" in country={country}" if country else "")
                           + (f", topic={topic}" if topic else ""),
                "results": [],
            })

        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity: similarity = 1 - (distance / 2)
        formatted_results = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            similarity = 1 - (dist / 2)

            # Skip results below confidence threshold
            if similarity < SIMILARITY_THRESHOLD:
                continue

            formatted_results.append({
                "text": doc,
                "similarity": round(similarity, 4),
                "source": {
                    "document_title": meta.get("document_title", "Unknown"),
                    "country": meta.get("country", "Unknown"),
                    "topic": meta.get("topic", ""),
                    "language": meta.get("language", ""),
                    "source_agency": meta.get("source_agency", ""),
                    "document_url": meta.get("document_url", ""),
                    "effective_date": meta.get("effective_date", ""),
                },
            })

        if not formatted_results:
            return json.dumps({
                "status": "low_confidence",
                "message": (
                    f"Found documents but none met the confidence threshold "
                    f"({SIMILARITY_THRESHOLD}). The knowledge base may not cover "
                    f"this topic yet. Query: '{query}'"
                ),
                "results": [],
                "best_distance": round(distances[0], 4) if distances else None,
            })

        return json.dumps({
            "status": "success",
            "query": query,
            "result_count": len(formatted_results),
            "results": formatted_results,
        })

    except Exception as e:
        logger.error(f"search_documents error: {e}", exc_info=True)
        return json.dumps({
            "status": "error",
            "message": f"Search failed: {str(e)}",
            "results": [],
        })


def _build_where_filter(
    country: str, topic: str
) -> Optional[dict]:
    """
    Build a ChromaDB `where` filter from country and topic.

    ChromaDB where syntax:
      Single condition:  {"country": "MY"}
      AND conditions:    {"$and": [{"country": "MY"}, {"topic": "social_security"}]}
    """
    conditions = []

    if country and country.strip():
        # Normalize: accept "Malaysia"/"malaysia"/"MY" etc.
        country_normalized = _normalize_country(country.strip())
        if country_normalized:
            conditions.append({"country": country_normalized})

    if topic and topic.strip():
        conditions.append({"topic": topic.strip().lower()})

    if len(conditions) == 0:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}


def _normalize_country(raw: str) -> str:
    """Map various country representations to the ISO codes used in metadata."""
    mapping = {
        # ISO codes (already correct)
        "MY": "MY", "ID": "ID", "PH": "PH", "TH": "TH", "ASEAN": "ASEAN",
        # Lowercase ISO
        "my": "MY", "id": "ID", "ph": "PH", "th": "TH", "asean": "ASEAN",
        # Full names
        "malaysia": "MY", "indonesia": "ID", "philippines": "PH",
        "thailand": "TH",
        # Common variants
        "filipino": "PH", "thai": "TH", "malay": "MY", "indonesian": "ID",
    }
    return mapping.get(raw, mapping.get(raw.lower(), raw.upper()))