#!/usr/bin/env python3
"""
preprocess_documents.py — AskAra+ RAG Document Cleaner
=======================================================
Fixes 5 scraper failure modes before chunking + ingest:

  1. Tab-widget content duplication  (same body repeated 4x under TAB: headers)
  2. Navigation boilerplate          (breadcrumbs, "KEMBALI KE RUANG UTAMA FAQ", etc.)
  3. Stub/empty files                (< MIN_WORDS useful words → skipped)
  4. Cross-file content overlap      (near-duplicate paragraphs across files)
  5. Over-fragmented tiny files      (handled by merge_groups below)

Usage (from project root or backend/):
    python preprocess_documents.py                         # clean in-place (backup made)
    python preprocess_documents.py --dry-run               # preview stats only
    python preprocess_documents.py --src path/to/docs      # custom input dir
    python preprocess_documents.py --out path/to/cleaned   # custom output dir
    python preprocess_documents.py --merge                 # also merge related files

Output:
    - Cleaned .txt files written to --out dir
    - Each file gets a report: words before → after, blocks removed
    - Files below MIN_WORDS threshold are logged but NOT written (skip list printed)
"""

import argparse
import hashlib
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────
MIN_WORDS        = 60      # files with fewer words after cleaning are stubs → skip
DEDUP_WINDOW     = 6       # paragraph similarity window for intra-file dedup
DEDUP_HASH_CHARS = 120     # how many chars to hash for paragraph fingerprinting
# ──────────────────────────────────────────────────────────────────────────────

# ── Boilerplate patterns to strip (exact line match OR regex) ─────────────────
STRIP_EXACT = {
    "KEMBALI KE RUANG UTAMA FAQ",
    "Kembali ke Penguatkuasaan",
    "Kembali ke Perkhidmatan",
    "SUMBER / SOURCE",
    "KEMBALI KE RUANG UTAMA",
    # JKM sidebar sections — appear in every scraped JKM article (77× duplicate)
    "DOKUMEN DAN PAUTAN BERKAITAN",
    # PERKESO portal link block — appears 23× across PERKESO docs
    "DOKUMEN PDF BERKAITAN",
    # JTKSM related docs sidebar
    "DOKUMEN BERKAITAN",
}

STRIP_REGEX = [
    # Breadcrumb trails:  Utama»Penguatkuasaan»Imigran-Imigran Larangan
    re.compile(r"^Utama\s*».*$", re.MULTILINE),
    # Source URL lines after "SUMBER / SOURCE"
    re.compile(r"^https?://\S+$", re.MULTILINE),
    # Date stamps from scraper:  November 3, 2021\n12:02 am
    re.compile(r"^\w+ \d{1,2}, \d{4}\s*$", re.MULTILINE),
    re.compile(r"^\d{1,2}:\d{2} (am|pm)\s*$", re.MULTILINE),
    # Empty download references
    re.compile(r"^\S+Download\s*$", re.MULTILINE),
    # "Mac 6, 2024" Malay date format
    re.compile(r"^(Januari|Februari|Mac|April|Mei|Jun|Julai|Ogos|September|Oktober|November|Disember)\s+\d{1,2},\s+\d{4}\s*$", re.MULTILINE),
    # JKM related-document link lines (bullet lines starting with "- " followed by a URL)
    re.compile(r"^-\s+.+:\s+https?://\S+$", re.MULTILINE),
    # EPF contribution form table header noise ("FOR THE MONTH" boilerplate — 71 occurrences)
    re.compile(r"^FOR THE MONTH\s*$", re.MULTILINE),
    # Kisah Kejayaan / success story links that appear as sidebar noise in JKM docs
    re.compile(r"^-\s*Kisah Kejayaan.*$", re.MULTILINE),
]

# ── TAB: header normalization ─────────────────────────────────────────────────
TAB_RE = re.compile(r"^TAB:\s*(.+)$", re.MULTILINE)


def tab_to_header(match: re.Match) -> str:
    """Convert   TAB: Anda tidak mempunyai pas   →   ANDA TIDAK MEMPUNYAI PAS"""
    label = match.group(1).strip()
    # Truncate very long tab labels (they're often full sentences)
    if len(label) > 80:
        label = label[:77] + "..."
    return "\n" + label.upper()


# ── Core cleaning function ────────────────────────────────────────────────────

def clean_text(raw: str) -> tuple[str, dict]:
    """
    Returns (cleaned_text, stats_dict).
    Stats: words_before, words_after, boilerplate_lines_removed, dedup_blocks_removed
    """
    stats = {
        "words_before": len(raw.split()),
        "boilerplate_removed": 0,
        "dedup_blocks_removed": 0,
    }

    # Step 1 — Normalize line endings
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Step 2 — Convert TAB: headers to ALL-CAPS section headers
    text = TAB_RE.sub(tab_to_header, text)

    # Step 3 — Strip regex patterns
    for pattern in STRIP_REGEX:
        before = text
        text = pattern.sub("", text)
        removed = before.count("\n") - text.count("\n")
        if removed > 0:
            stats["boilerplate_removed"] += removed

    # Step 4 — Strip exact-match lines (after stripping, count removals)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped in STRIP_EXACT:
            stats["boilerplate_removed"] += 1
        else:
            cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # Step 5 — Collapse 3+ consecutive blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Step 6 — Intra-file paragraph deduplication
    # Split on blank lines, fingerprint each paragraph, skip duplicates
    paragraphs = text.split("\n\n")
    seen_hashes: set[str] = set()
    unique_paragraphs = []

    for para in paragraphs:
        stripped_para = para.strip()
        if not stripped_para:
            continue

        # Fingerprint: first DEDUP_HASH_CHARS chars, lowercased, whitespace-normalized
        fp_text = " ".join(stripped_para.lower().split())[:DEDUP_HASH_CHARS]
        fp = hashlib.md5(fp_text.encode()).hexdigest()

        if fp in seen_hashes:
            stats["dedup_blocks_removed"] += 1
        else:
            seen_hashes.add(fp)
            unique_paragraphs.append(para)

    text = "\n\n".join(unique_paragraphs)

    # Step 7 — Strip leading/trailing whitespace
    text = text.strip()

    stats["words_after"] = len(text.split())
    return text, stats


# ── Merge groups ──────────────────────────────────────────────────────────────
# Define which source files merge into which output file.
# Keys = output filename stem. Values = list of input filename stems (prefix match ok).
# Add more groups here as you identify related files.

MERGE_GROUPS: dict[str, list[str]] = {
    # ── JIM (Jabatan Imigresen Malaysia) ────────────────────────────────────
    "MY_IMI_PenguatKuasaan": [
        "MY_IMI_penguatkuasaan",
        "MY_IMI_penguatkuasaan_imigran-imigran-larangan",
        "MY_IMI_penguatkuasaan_hilang-pasport-dokumen-perjalanan",
        "MY_IMI_faq_bpdtjim_bahagian-pengurusan-depot-dan-tahanan",
        "MY_IMI_perkhidmatan-utama_tindakan-kompaun",
    ],
    "MY_IMI_PasLawatanKerja": [
        "MY_IMI_faq_bpajim_bahagian-pekerja-asing",
        "MY_IMI_faq-umum_umum-2",
        "MY_IMI_perkhidmatan-utama_pas_pas-lawatan_pas-lawatan-kerja",
        "MY_IMI_perkhidmatan-utama_pas_pas-lawatan_pas-lawatan-kerja-sementara",
        "MY_IMI_perkhidmatan-utama_pas_pas-penggajian",
        "MY_IMI_perkhidmatan-utama_pas_expatriat",
        "MY_IMI_perkhidmatan-utama_pas_pas-penggajian-2",
    ],
    "MY_IMI_VisaDanPermit": [
        "MY_IMI_perkhidmatan-utama_visa_keperluan-visa",
        "MY_IMI_perkhidmatan-utama_visa_kadar-bayaran-visa",
        "MY_IMI_perkhidmatan-utama_visa_visa-dengan-rujukan-vdr",
        "MY_IMI_perkhidmatan-utama_permit-masuk",
    ],
    "MY_IMI_PasSosialPelajarResiden": [
        "MY_IMI_perkhidmatan-utama_pas_pas-lawatan_pas-lawatan-sosial-jangka-pan",
        "MY_IMI_perkhidmatan-utama_pas_pas-pelajar",
        "MY_IMI_perkhidmatan-utama_pas_pas-residen",
        "MY_IMI_perkhidmatan-utama_pas_pas-lawatan-ikhtisas",
    ],
    "MY_IMI_DokumenPerjalanan": [
        "MY_IMI_perkhidmatan-utama_dokumen-perjalanan-terhad",
        "MY_IMI_perkhidmatan-utama_dokumen-gantian-perjalanan",
        "MY_IMI_perkhidmatan-utama_pas-sempadan-lintas-batas",
    ],
    "MY_IMI_SyaratKemasukanMalaysia": [
        "MY_IMI_perkhidmatan-utama_syarat-kemasukan-ke-malaysia",
        "MY_IMI_perkhidmatan-utama_syarat-kemasukan_warganegara-asing",
        "MY_IMI_perkhidmatan-utama_syarat-kemasukan_pemastautin-tetap",
        "MY_IMI_perkhidmatan-utama_syarat-kemasukan_kanak-kanak",
        "MY_IMI_perkhidmatan-utama_syarat-kemasukan_kesalahan-sering-dilakukan",
        "MY_IMI_perkhidmatan-utama_tatacara-memohon-endorsemen-ke",
    ],
    "MY_IMI_PembantuRumahAsing": [
        "MY_IMI_perkhidmatan-utama_pembantu-rumah-asing",
    ],

    # ── PERKESO / SOCSO ──────────────────────────────────────────────────────
    "MY_PERKESO_PerlindunganPekerjaAsing": [
        "MY_PERKESO_perlindungan_pekerja-asing",
        "MY_PERKESO_perlindungan_pekerja-bermajikan",
        "MY_PERKESO_perlindungan_pekerja-domestik",
        "MY_PERKESO_perlindungan_pekerja-asing-2",
    ],
    "MY_PERKESO_CarumanDanPendaftaran": [
        "MY_PERKESO_majikan-pekerja_caruman",
        "MY_PERKESO_majikan-pekerja_kadar-caruman",
        "MY_PERKESO_majikan-pekerja_pembayaran",
        "MY_PERKESO_majikan-pekerja_pendaftaran-majikan",
        "MY_PERKESO_majikan-pekerja_penguatkuasaan",
    ],
    "MY_PERKESO_FaedahDanPermohonan": [
        "MY_PERKESO_perlindungan_permohonan-faedah",
        "MY_PERKESO_perlindungan_insurans-pekerjaan",
        "MY_PERKESO_perlindungan_skim-bencana-pekerjaan",
        "MY_PERKESO_perlindungan_skim-keselamatan-sosial-pekerja-asing",
    ],
    "MY_PERKESO_PerubatanDanPemulihan": [
        "MY_PERKESO_perubatan_panel-perubatan",
        "MY_PERKESO_perubatan_rawatan-perubatan",
        "MY_PERKESO_perubatan_pemulihan-vokasional",
        "MY_PERKESO_perubatan_rawatan-perubatan-luar-negara",
    ],

    # ── JTKSM (Jabatan Tenaga Kerja Semenanjung Malaysia) ───────────────────
    "MY_JTKSM_AduanBuruh": [
        "MY_JTKSM_perkhidmatan_aduan",
        "MY_JTKSM_perkhidmatan_aduan_akta-panduan",
        "MY_JTKSM_soalan-lazim_soalan-lazim-kes-buruh",
        "MY_JTKSM_perkhidmatan_kes-buruh",
        "MY_JTKSM_perkhidmatan_kes-buruh_akta-panduan",
    ],
    "MY_JTKSM_PemberhentianPekerja": [
        "MY_JTKSM_perkhidmatan_pemberhentian-pekerja",
        "MY_JTKSM_perkhidmatan_pemberhentian-pekerja_akta-panduan",
        "MY_JTKSM_perkhidmatan_pemberhentian-pekerja_borang-pemberhen",
        "MY_JTKSM_soalan-lazim_soalan-lazim-pemberhentian-pekerja",
    ],
    "MY_JTKSM_PenggajianPekerjaAsing": [
        "MY_JTKSM_perkhidmatan_penggajian-pekerja-asing",
        "MY_JTKSM_perkhidmatan_penggajian-pekerja-asing-0",
        "MY_JTKSM_perkhidmatan_penggajian-pekerja-asing_borang-pengga",
        "MY_JTKSM_perkhidmatan_penggajian-pekerja-asing_soalan-lazim-",
        "MY_JTKSM_perkhidmatan_penggajian-pekerja-asing_akta-panduan",
    ],
    "MY_JTKSM_PermitPerburuhan": [
        "MY_JTKSM_perkhidmatan_permit-perburuhan",
        "MY_JTKSM_perkhidmatan_permit-perburuhan_akta-panduan",
        "MY_JTKSM_soalan-lazim_soalan-lazim-permit-perburuhan",
    ],
    "MY_JTKSM_GajiDanInstrumenPembayaran": [
        "MY_JTKSM_perkhidmatan_instrumen-pembayaran-gaji",
        "MY_JTKSM_perkhidmatan_instrumen-pembayaran-gaji_akta-panduan",
        "MY_JTKSM_soalan-lazim_soalan-lazim-instrumen-pembayaran",
    ],
    "MY_JTKSM_PerumahanPekerja": [
        "MY_JTKSM_perkhidmatan_perumahan-pekerja",
        "MY_JTKSM_perkhidmatan_perumahan-pekerja-0",
        "MY_JTKSM_perkhidmatan_perumahan-pekerja-1",
        "MY_JTKSM_soalan-lazim_perumahan-penginapan-dan-kemudaha",
        "MY_JTKSM_soalan-lazim_soalan-lazim-perumahan-pekerja",
    ],
    "MY_JTKSM_AgensiPekerjaanSwasta": [
        "MY_JTKSM_perkhidmatan_agensi-pekerjaan-swasta",
        "MY_JTKSM_perkhidmatan_agensi-pekerjaan-swasta_akta-panduan",
        "MY_JTKSM_soalan-lazim_soalan-lazim-agensi-pekerjaan-swasta",
    ],
    "MY_JTKSM_PendaftaranTempat": [
        "MY_JTKSM_perkhidmatan_pendaftaran-tempat-pekerjaan",
        "MY_JTKSM_perkhidmatan_pendaftaran-tempat-pekerjaan_akta-pand",
    ],

    # ── JKM (Jabatan Kebajikan Masyarakat) ───────────────────────────────────
    "MY_JKM_BantuanBencana": [
        "MY_JKM_pengurusan-bencana",
        "MY_JKM_main_article_bantuan-bencana",
        "MY_JKM_main_article_bantuan-wang-ihsan",
        "MY_JKM_main_article_pusat-pemindahan-sementara",
        "MY_JKM_main_article_pengurusan-bencana",
        "MY_JKM_main_article_garis-panduan-pengurusan-keselamatan-di-",
    ],
    "MY_JKM_BantuanKebajikan": [
        "MY_JKM_skim-bantuan-kebajikan",
        "MY_JKM_main_article_bantuan-bulanan",
        "MY_JKM_main_article_bantuan-sekaligus",
        "MY_JKM_main_article_pengenalan-skim-bantuan-kebajikan",
        "MY_JKM_main_article_skim-bantuan-kebajikan",
    ],
    "MY_JKM_BantuanKecemasan": [
        "MY_JKM_main_article_bantuan-kecemasan-individu",
        "MY_JKM_main_article_bantuan-khas-perbendaharaan",
        "MY_JKM_main_article_2-years-exit-programme-2yep",
    ],
    "MY_JKM_PerlindunganATIP": [
        "MY_JKM_perlindungan-mangsa-atip",
        "MY_JKM_main_article_perlindungan-mangsa-atip",
        "MY_JKM_main_article_mangsa-pemerdagangan-orang",
        "MY_JKM_main_article_institusi-kebajikan-rumah-perlindungan",
        "MY_JKM_main_article_rumah-perlindungan",
    ],
    "MY_JKM_OrangPapa": [
        "MY_JKM_orang-papa",
        "MY_JKM_main_article_orang-papa",
        "MY_JKM_main_article_rumah-kebajikan-orang-papa",
        "MY_JKM_main_article_permohonan-masuk-ke-rth",
    ],
    "MY_JKM_KanakKanak": [
        "MY_JKM_kanak---kanak",
        "MY_JKM_main_article_pasukan-pelindungan-kanak-kanak",
        "MY_JKM_main_article_perkembangan-awal-kanak-kanak",
        "MY_JKM_main_article_aduan-kes-penderaan-pengabaian-kanak-kan",
        "MY_JKM_main_article_permohonan-pengambilan-kanak-kanak-seksy",
        "MY_JKM_main_article_jabatan-pembangunan-kanak-kanak",
    ],
    "MY_JKM_OrangKurangUpaya": [
        "MY_JKM_main_article_pendaftaran-orang-kurang-upaya-oku",
        "MY_JKM_main_article_jabatan-pembangunan-orang-kurang-upaya",
        "MY_JKM_main_article_taska-oku",
        "MY_JKM_main_article_semakan-edkk",
        "MY_JKM_main_article_perkhidmatan-institusi-bengkel-daya",
        "MY_JKM_main_article_program-pemulihan-dalam-komuniti-pdk",
    ],
    "MY_JKM_PerkhidmatanSosial": [
        "MY_JKM_perkhidmatan-kerja-sosial",
        "MY_JKM_psikologi-dan-kaunseling",
        "MY_JKM_perintah-khidmat-masyarakat",
        "MY_JKM_main_article_orang-awam",
        "MY_JKM_main_article_permohonan-pengiktirafan-sebagai-jurulat",
    ],
    "MY_JKM_InstitusiKebajikan": [
        "MY_JKM_institusi-kebajikan",
        "MY_JKM_main_article_permohonan-kemasukan-ke-pusat-jagaan-sin",
        "MY_JKM_main_article_perkhidmatan-respite-care-rumah-seri-ken",
        "MY_JKM_main_article_program-pemulihan-dan-penginsafan-pelati",
        "MY_JKM_main_article_garis-panduan-penubuhan-dan-pendaftaran-",
        "MY_JKM_main_article_modul-dan-peraturan-berkaitan-pkm",
    ],
}

# Files to skip outright (exact stem match). These are stubs or irrelevant.
SKIP_STEMS: set[str] = {
    "MY_JKM_warga-jkm",
    "MY_JKM_badan-ngo",
    "MY_JTKSM_kalkulator_gaji",
    "MY_IMI_perkhidmatan-utama_visa",
    "MY_IMI_perkhidmatan-utama_pas",
    "MY_IMI_perkhidmatan-utama",
    # AI-generated files (replace with real scraped data):
    "MY_IMI_PLKS_Renewal_Guide",
    "MY_IMI_WorkPass_Types",
    "MY_IMI_Overstay_Amnesty",
    "MY_IMI_Detained_Rights",
}

# Patterns in stem that indicate internal/org pages not useful for citizens
SKIP_STEM_PATTERNS: list[re.Pattern] = [
    # JKM internal org pages (NOT citizen-facing). The prefix "main_article_bahagian"
    # is specific to JKM's CMS. IMI FAQ pages like "faq_bpajim_bahagian-pekerja-asing"
    # ARE citizen-facing and must NOT be skipped.
    re.compile(r"main_article_bahagian-"),   # JKM internal department pages
    re.compile(r"pengurusan-tertinggi"),     # senior management page
    re.compile(r"majlis-(penasihat|kebangsaan)"),
    re.compile(r"jawatankuasa"),
    re.compile(r"sekretariat"),
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-"),  # UUID-named mystery articles
]


def should_skip_stem(stem: str) -> Optional[str]:
    """Returns reason string if file should be skipped, else None."""
    if stem in SKIP_STEMS:
        return "in skip list"
    for pat in SKIP_STEM_PATTERNS:
        if pat.search(stem):
            return f"matches skip pattern '{pat.pattern}'"
    return None


# ── Meta.json handling ────────────────────────────────────────────────────────

def load_meta(meta_path: Path) -> Optional[dict]:
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"  Cannot read {meta_path.name}: {e}")
        return None


def merge_metas(metas: list[dict], output_stem: str) -> dict:
    """Combine multiple meta.json dicts into one for a merged file."""
    if not metas:
        return {}
    base = metas[0].copy()
    # Collect all source URLs
    urls = [m.get("document_url", "") for m in metas if m.get("document_url")]
    base["document_url"] = urls[0] if len(urls) == 1 else urls
    # Use the output stem as the document title
    base["document_title"] = output_stem.replace("_", " ").replace("-", " ")
    # Use the latest effective date
    dates = [m.get("effective_date", "") for m in metas if m.get("effective_date")]
    if dates:
        base["effective_date"] = max(dates)
    return base


# ── File processing ───────────────────────────────────────────────────────────

def process_file(
    src: Path,
    out_dir: Path,
    dry_run: bool = False,
) -> Optional[dict]:
    """
    Clean a single .txt file. Returns report dict or None if skipped.
    If not dry_run, writes cleaned file to out_dir.
    """
    stem = src.stem
    meta_path = src.parent / (stem + ".meta.json")

    reason = should_skip_stem(stem)
    if reason:
        return {"file": stem, "action": "SKIP", "reason": reason}

    raw = src.read_text(encoding="utf-8")
    cleaned, stats = clean_text(raw)

    # Check word count
    if stats["words_after"] < MIN_WORDS:
        return {
            "file": stem,
            "action": "STUB",
            "reason": f"only {stats['words_after']} words after cleaning (min={MIN_WORDS})",
            "stats": stats,
        }

    # Write output
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / src.name).write_text(cleaned, encoding="utf-8")
        # Copy meta.json if it exists
        if meta_path.exists():
            shutil.copy2(meta_path, out_dir / meta_path.name)

    return {
        "file": stem,
        "action": "CLEAN",
        "stats": stats,
        "delta_words": stats["words_before"] - stats["words_after"],
        "boilerplate_removed": stats["boilerplate_removed"],
        "dedup_removed": stats["dedup_blocks_removed"],
    }


def process_merge_group(
    group_stem: str,
    member_stems: list[str],
    src_dir: Path,
    out_dir: Path,
    dry_run: bool = False,
) -> dict:
    """
    Merge multiple cleaned files into one. Returns report dict.
    Expects cleaned files already written to out_dir (or src_dir for dry_run).
    """
    merged_parts = []
    collected_metas = []
    found_stems = []
    seen_paths: set = set()  # prevent double-ingesting the same file

    for member_stem in member_stems:
        # Prefer exact match; only fall back to prefix glob when exact doesn't exist
        exact = src_dir / f"{member_stem}.txt"
        candidates = [exact] if exact.exists() else list(src_dir.glob(f"{member_stem}*.txt"))
        if not candidates:
            log.debug(f"  Merge group '{group_stem}': no file found for '{member_stem}'")
            continue

        for candidate in candidates:
            if candidate in seen_paths:
                continue
            seen_paths.add(candidate)
            if candidate.stem in SKIP_STEMS:
                continue
            raw = candidate.read_text(encoding="utf-8")
            cleaned, stats = clean_text(raw)
            if stats["words_after"] < MIN_WORDS:
                log.debug(f"  Merge group '{group_stem}': '{candidate.stem}' is stub, skipping")
                continue
            merged_parts.append(f"{'='*60}\n{candidate.stem}\n{'='*60}\n\n{cleaned}")
            found_stems.append(candidate.stem)

            # Load meta
            meta_path = candidate.parent / (candidate.stem + ".meta.json")
            if meta_path.exists():
                m = load_meta(meta_path)
                if m:
                    collected_metas.append(m)

    if not merged_parts:
        return {"group": group_stem, "action": "EMPTY", "members": member_stems}

    merged_text = "\n\n\n".join(merged_parts)
    merged_meta = merge_metas(collected_metas, group_stem)

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{group_stem}.txt"
        out_path.write_text(merged_text, encoding="utf-8")
        if merged_meta:
            meta_out = out_dir / f"{group_stem}.meta.json"
            with open(meta_out, "w", encoding="utf-8") as f:
                json.dump(merged_meta, f, ensure_ascii=False, indent=4)

    return {
        "group": group_stem,
        "action": "MERGE",
        "members_found": found_stems,
        "members_requested": member_stems,
        "total_words": len(merged_text.split()),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clean and optionally merge AskAra+ scraped documents")
    parser.add_argument("--src",      default="../data/documents", help="Source documents dir")
    parser.add_argument("--out",      default="../data/documents_clean", help="Output dir for cleaned files")
    parser.add_argument("--dry-run",  action="store_true", help="Preview only, don't write files")
    parser.add_argument("--merge",    action="store_true", help="Also merge related files per MERGE_GROUPS")
    parser.add_argument("--in-place", action="store_true", help="Write cleaned files back to --src (backup first)")
    args = parser.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)

    if not src_dir.exists():
        log.error(f"Source dir not found: {src_dir}")
        return

    if args.in_place:
        out_dir = src_dir
        if not args.dry_run:
            backup_dir = src_dir.parent / (src_dir.name + "_backup")
            if not backup_dir.exists():
                log.info(f"Backing up {src_dir} → {backup_dir}")
                shutil.copytree(src_dir, backup_dir)

    mode = "DRY RUN — " if args.dry_run else ""
    log.info(f"{mode}Source: {src_dir}")
    log.info(f"{mode}Output: {out_dir}")

    txt_files = sorted(src_dir.glob("MY_*.txt")) + sorted(src_dir.glob("ASEAN_*.txt"))
    log.info(f"Found {len(txt_files)} .txt files to process")

    reports = []
    for f in txt_files:
        report = process_file(f, out_dir, dry_run=args.dry_run)
        if report:
            reports.append(report)

    # ── Print summary ──────────────────────────────────────────────────────
    clean   = [r for r in reports if r["action"] == "CLEAN"]
    stubs   = [r for r in reports if r["action"] == "STUB"]
    skipped = [r for r in reports if r["action"] == "SKIP"]

    print("\n" + "="*64)
    print(f"  PREPROCESSING REPORT  ({mode.strip()})")
    print("="*64)
    print(f"  ✅ Cleaned : {len(clean):>4} files")
    print(f"  ⚠️  Stubs   : {len(stubs):>4} files (< {MIN_WORDS} words after clean)")
    print(f"  🚫 Skipped : {len(skipped):>4} files (irrelevant/AI-generated)")
    print()

    if clean:
        total_words_removed = sum(r.get("delta_words", 0) for r in clean)
        total_dedup = sum(r.get("dedup_removed", 0) for r in clean)
        total_boilerplate = sum(r.get("boilerplate_removed", 0) for r in clean)
        print(f"  Total words removed   : {total_words_removed:,}")
        print(f"  Boilerplate lines cut : {total_boilerplate:,}")
        print(f"  Duplicate blocks cut  : {total_dedup:,}")
        print()

        # Show biggest wins
        top_cleaned = sorted(clean, key=lambda r: r.get("delta_words", 0), reverse=True)[:8]
        print("  Top files by words removed:")
        for r in top_cleaned:
            s = r["stats"]
            print(f"    {r['file'][:55]:<55} "
                  f"{s['words_before']:>5}w → {s['words_after']:>5}w  "
                  f"(-{r['delta_words']:>4}w, "
                  f"{r['dedup_removed']}×dedup, "
                  f"{r['boilerplate_removed']}×boilerplate)")

    if stubs:
        print("\n  Stub files (will NOT be ingested):")
        for r in stubs:
            print(f"    {r['file'][:60]:<60} — {r['reason']}")

    if skipped:
        print("\n  Skipped files:")
        for r in skipped:
            print(f"    {r['file'][:60]:<60} — {r['reason']}")

    # ── Merge phase ────────────────────────────────────────────────────────
    if args.merge and MERGE_GROUPS:
        print("\n" + "="*64)
        print("  MERGE PHASE")
        print("="*64)
        merge_src = out_dir if not args.dry_run else src_dir
        merge_out = out_dir

        for group_stem, members in MERGE_GROUPS.items():
            result = process_merge_group(group_stem, members, merge_src, merge_out, dry_run=args.dry_run)
            action = result["action"]
            if action == "MERGE":
                found = result["members_found"]
                total_w = result["total_words"]
                print(f"  ✅ MERGE → {group_stem}.txt  ({len(found)} files, {total_w:,}w)")
                for m in found:
                    print(f"       + {m}")
                # ── Remove component files from out_dir to prevent duplicate chunks ──
                if not args.dry_run:
                    for stem in found:
                        for ext in (".txt", ".meta.json"):
                            component = out_dir / f"{stem}{ext}"
                            if component.exists():
                                component.unlink()
                                log.debug(f"  Removed component: {component.name}")
            elif action == "EMPTY":
                print(f"  ⚠️  EMPTY  {group_stem} — no usable members found")

    print("\n" + "="*64)
    if args.dry_run:
        print("  DRY RUN complete — no files written.")
        print("  Re-run without --dry-run to write cleaned files.")
    else:
        print(f"  Done. Cleaned files written to: {out_dir}")
        print("  Next steps:")
        print("    1. Review output dir, spot-check a few files")
        print("    2. Copy .meta.json files for any that don't have them yet")
        print("    3. uv run python load_chromadb.py --clear  (re-ingest)")
    print("="*64 + "\n")


if __name__ == "__main__":
    main()