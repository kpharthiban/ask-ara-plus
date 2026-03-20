# AskAra+ — Document Loading Guide (for Lineysha)

## Quick Start

```bash
# 1. Install dependencies
cd backend
uv sync

# 2. Test ChromaDB is working
uv run python db.py

# 3. Put your .txt + .meta.json files in data/documents/

# 4. Chunk them (no dependencies needed — runs standalone)
python ../data/scripts/chunk_documents.py

# 5. Load into ChromaDB (run from backend/ so it picks up .venv)
uv run python ../data/scripts/load_chromadb.py

# 6. Verify retrieval works
uv run python ../data/scripts/test_retrieval.py
```

> **Note:** Steps 5 and 6 must be run from the `backend/` folder because
> they depend on `chromadb` and `sentence-transformers` which live in
> `backend/.venv`. Step 4 is pure Python — run it from anywhere.

## File Naming Convention

Each document needs TWO files:
```
data/documents/
├── MY_socso_guide.txt           ← extracted text
├── MY_socso_guide.meta.json     ← metadata (MUST match .txt filename)
├── ID_uu18_migrant.txt
├── ID_uu18_migrant.meta.json
├── PH_owwa_benefits.txt
├── PH_owwa_benefits.meta.json
└── ...
```

Name format: `{COUNTRY}_{short_name}.txt`

## Meta.json Template

Copy this for each document and fill in the values:

```json
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
```

### Valid `country` values:
- `MY` — Malaysia
- `ID` — Indonesia
- `PH` — Philippines
- `TH` — Thailand
- `ASEAN` — ASEAN-level documents

### Valid `topic` values:
- `worker_rights` — employment laws, labour protection
- `migrant_rights` — migrant worker specific
- `social_security` — SOCSO, BPJS, SSS, etc.
- `disaster_relief` — flood aid, BLT, emergency assistance
- `health` — healthcare access, insurance
- `livelihood` — DTI programs, skills training
- `general` — cross-cutting / multi-topic

### Valid `language` values:
- `ms` — Malay
- `id` — Indonesian
- `en` — English
- `tl` — Filipino/Tagalog
- `th` — Thai

## Chunking Details

The chunking script automatically:
- Detects section headings (Bab, Pasal, Section, Bahagian, มาตรา, etc.)
- Creates ~512 token chunks with 50-token overlap
- Prefixes each chunk with `[Section Heading]` for context
- Generates unique IDs: `{COUNTRY}_{filename}_{001}`
- Computes content hashes for versioning

## Re-loading After Adding More Documents

Just run steps 4–6 again. The loader uses **upsert** — existing chunks
with the same ID get updated, new ones get added. No duplicates.

To do a clean reload (from `backend/`):
```bash
uv run python ../data/scripts/load_chromadb.py --clear
```

## Troubleshooting

**"No .meta.json found"** → The script still works but uses default metadata.
Create the .meta.json file for better retrieval quality.

**Low similarity scores in test** → Check that the .txt file is clean
(no PDF artefacts, headers/footers removed, encoding is UTF-8).

**ChromaDB permission error** → Make sure `data/chromadb/` directory exists
and is writable.