"""
AskAra+ — Document Chunking Script

Reads .txt files from data/documents/, splits them into ~512-token chunks
with 50-token overlap, and outputs JSON ready for ChromaDB loading.

HOW TO USE (for Lineysha):
──────────────────────────
1. Place your extracted .txt files in  data/documents/
2. Place matching .meta.json files next to each .txt file
   (same name, e.g. MY_socso_guide.txt + MY_socso_guide.meta.json)
3. Run:  python data/scripts/chunk_documents.py
4. Output goes to:  data/documents/chunks.json

META.JSON FORMAT (one per document):
{
    "country": "MY",
    "topic": "worker_rights",
    "language": "ms",
    "source_agency": "PERKESO",
    "document_title": "Panduan Pendaftaran SOCSO",
    "effective_date": "2024-01-01",
    "expiry_date": null,
    "document_url": "https://perkeso.gov.my/..."
}

The script will automatically:
- Detect section headings (Bab, Section, Pasal, numbered headings, etc.)
- Chunk by ~512 tokens with 50-token overlap
- Prefix each chunk with its parent section heading
- Generate unique chunk IDs
- Compute a content hash for versioning
"""

import json
import hashlib
import os
import re
import glob
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "documents")
OUTPUT_FILE = os.path.join(DOCUMENTS_DIR, "chunks.json")
CHUNK_SIZE_TOKENS = 512
OVERLAP_TOKENS = 50

# Rough token estimation: 1 token ≈ 4 characters for Latin scripts,
# ~1.5 chars for CJK/Thai. We use a conservative 3.5 average for SEA text.
CHARS_PER_TOKEN = 3.5

# ---------------------------------------------------------------------------
# Section heading patterns (ASEAN government docs)
# ---------------------------------------------------------------------------
HEADING_PATTERNS = [
    # Indonesian: Bab I, Bab II, Pasal 1, etc.
    r"^(Bab\s+[IVXLCDM\d]+\.?.*)",
    r"^(Pasal\s+\d+\.?.*)",
    r"^(BAB\s+[IVXLCDM\d]+\.?.*)",
    # Malay: Bahagian, Seksyen, Peraturan
    r"^(Bahagian\s+[IVXLCDM\d]+\.?.*)",
    r"^(Seksyen\s+\d+\.?.*)",
    r"^(Peraturan\s+\d+\.?.*)",
    # English: Section, Part, Chapter, Article
    r"^(Section\s+\d+\.?.*)",
    r"^(Part\s+[IVXLCDM\d]+\.?.*)",
    r"^(Chapter\s+\d+\.?.*)",
    r"^(Article\s+\d+\.?.*)",
    # Thai: มาตรา (Section), บท (Chapter) — common in Thai legal docs
    r"^(มาตรา\s*\d+\.?.*)",
    r"^(บทที่\s*\d+\.?.*)",
    # Filipino: Seksyon, Artikulo
    r"^(Seksyon\s+\d+\.?.*)",
    r"^(Artikulo\s+\d+\.?.*)",
    # Generic numbered headings: "1.", "1.1", "1.1.1"
    r"^(\d+\.\d*\s+[A-Z\u0E00-\u0E7F].*)",
    # ALL-CAPS lines (often headings in gov docs), min 4 chars
    r"^([A-Z][A-Z\s]{3,}[A-Z])$",
]


def is_heading(line: str) -> bool:
    """Check if a line matches any known heading pattern."""
    stripped = line.strip()
    if not stripped:
        return False
    for pattern in HEADING_PATTERNS:
        if re.match(pattern, stripped):
            return True
    return False


def estimate_tokens(text: str) -> int:
    """Rough token count estimate."""
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def content_hash(text: str) -> str:
    """SHA-256 hash of chunk content for versioning."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------

def split_into_sections(text: str) -> list[dict]:
    """Split document text into sections based on heading detection.

    Returns list of {"heading": str, "body": str}.
    """
    lines = text.split("\n")
    sections: list[dict] = []
    current_heading = "Introduction"
    current_body_lines: list[str] = []

    for line in lines:
        if is_heading(line):
            # Save previous section
            body = "\n".join(current_body_lines).strip()
            if body:
                sections.append({
                    "heading": current_heading,
                    "body": body,
                })
            current_heading = line.strip()
            current_body_lines = []
        else:
            current_body_lines.append(line)

    # Don't forget the last section
    body = "\n".join(current_body_lines).strip()
    if body:
        sections.append({
            "heading": current_heading,
            "body": body,
        })

    return sections


def chunk_section(heading: str, body: str) -> list[str]:
    """Split a section body into overlapping chunks of ~CHUNK_SIZE_TOKENS.

    Each chunk is prefixed with the section heading for context.
    """
    prefix = f"[{heading}]\n"
    prefix_tokens = estimate_tokens(prefix)
    available_tokens = CHUNK_SIZE_TOKENS - prefix_tokens

    # Split body into paragraphs first (preserve natural boundaries)
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]

    # If no paragraph breaks, split by sentences
    if len(paragraphs) == 1:
        # Simple sentence splitter (handles ., ?, ! followed by space)
        sentences = re.split(r"(?<=[.!?])\s+", paragraphs[0])
        paragraphs = sentences

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        # If a single paragraph exceeds available tokens, hard-split it
        if para_tokens > available_tokens:
            # Flush current buffer
            if current_parts:
                chunks.append(prefix + " ".join(current_parts))
                # Keep overlap
                overlap_text = " ".join(current_parts)
                overlap_chars = int(OVERLAP_TOKENS * CHARS_PER_TOKEN)
                overlap_part = overlap_text[-overlap_chars:] if len(overlap_text) > overlap_chars else overlap_text
                current_parts = [overlap_part] if overlap_part.strip() else []
                current_tokens = estimate_tokens(overlap_part) if current_parts else 0

            # Hard-split the large paragraph by character count
            chunk_chars = int(available_tokens * CHARS_PER_TOKEN)
            overlap_chars = int(OVERLAP_TOKENS * CHARS_PER_TOKEN)
            start = 0
            while start < len(para):
                end = start + chunk_chars
                chunk_text = para[start:end]
                chunks.append(prefix + chunk_text)
                start = end - overlap_chars
            continue

        # Normal case: accumulate paragraphs
        if current_tokens + para_tokens > available_tokens and current_parts:
            # Flush
            chunks.append(prefix + " ".join(current_parts))
            # Keep overlap from end of current buffer
            overlap_text = " ".join(current_parts)
            overlap_chars = int(OVERLAP_TOKENS * CHARS_PER_TOKEN)
            overlap_part = overlap_text[-overlap_chars:] if len(overlap_text) > overlap_chars else overlap_text
            current_parts = [overlap_part] if overlap_part.strip() else []
            current_tokens = estimate_tokens(overlap_part) if current_parts else 0

        current_parts.append(para)
        current_tokens += para_tokens

    # Final flush
    if current_parts:
        chunks.append(prefix + " ".join(current_parts))

    return chunks


def chunk_document(filepath: str, metadata: dict) -> list[dict]:
    """Chunk a single document file into ChromaDB-ready records.

    Returns list of:
    {
        "id": "MY_socso_guide_001",
        "document": "chunk text...",
        "metadata": { ... }
    }
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    filename = Path(filepath).stem
    country = metadata.get("country", "XX").upper()

    sections = split_into_sections(text)
    all_chunks: list[dict] = []
    chunk_idx = 0

    for section in sections:
        chunks = chunk_section(section["heading"], section["body"])
        for chunk_text in chunks:
            chunk_idx += 1
            chunk_id = f"{country}_{filename}_{chunk_idx:03d}"
            chunk_meta = {
                **metadata,
                "section_heading": section["heading"],
                "chunk_hash": content_hash(chunk_text),
            }
            all_chunks.append({
                "id": chunk_id,
                "document": chunk_text,
                "metadata": chunk_meta,
            })

    return all_chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    docs_dir = os.path.abspath(DOCUMENTS_DIR)
    print(f"[Chunker] Scanning: {docs_dir}")

    txt_files = glob.glob(os.path.join(docs_dir, "**", "*.txt"), recursive=True)
    # Exclude chunks.json artefacts
    txt_files = [f for f in txt_files if not f.endswith("chunks.json")]

    if not txt_files:
        print("[Chunker] No .txt files found in data/documents/")
        print("          Place your extracted documents there and re-run.")
        return

    all_chunks: list[dict] = []

    for txt_path in sorted(txt_files):
        # Look for matching .meta.json
        meta_path = txt_path.replace(".txt", ".meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            # Auto-detect what we can from filename
            filename = Path(txt_path).stem
            parts = filename.split("_", 1)
            metadata = {
                "country": parts[0].upper() if len(parts) > 1 else "XX",
                "topic": "general",
                "language": "unknown",
                "source_agency": "unknown",
                "document_title": filename,
                "effective_date": None,
                "expiry_date": None,
                "document_url": None,
            }
            print(f"  ⚠ No .meta.json for {Path(txt_path).name} — using defaults")

        chunks = chunk_document(txt_path, metadata)
        all_chunks.extend(chunks)
        print(f"  ✓ {Path(txt_path).name}  →  {len(chunks)} chunks")

    # Write output
    output_path = os.path.abspath(OUTPUT_FILE)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n[Chunker] Done! {len(all_chunks)} total chunks → {output_path}")
    print(f"[Chunker] Next step: python data/scripts/load_chromadb.py")


if __name__ == "__main__":
    main()