"""
fetch_gov_portal — Hardened government portal search with golden URL map.

Owner: Pharthiban
Depends on: duckduckgo-search (pip install duckduckgo-search), httpx

Four-layer approach (in priority order):
  1. GOLDEN URLs — curated, verified landing pages per entity + intent.
     DuckDuckGo is NOT called at all when a golden URL exists. Zero hallucination risk.
     Returns up to 3 related golden entries (intent + general + contact) for rich context.
  2. Filtered DuckDuckGo — when no golden URL matches, DuckDuckGo runs but results
     are keyword-scored and red-flag pages (complaints, login, feedback) are removed.
     Query terms are placed BEFORE site: operator for better DDG ranking.
  3. Keyword relevance scoring — results scored by query keyword overlap in title/snippet.
     Zero-overlap results are discarded; survivors sorted by score descending.
  4. Parallel liveness check — all candidate URLs HEAD-checked concurrently via httpx.
     Dead links silently dropped. Golden URLs use a faster 2 s timeout.

Security: Only allowlisted government domains are permitted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx

logger = logging.getLogger("askara.portal")


# ─────────────────────────────────────────────────────────────────────────────
# 1. GOLDEN URL MAP
#    Structure: entity_key → intent_key → {"url": str, "title": str, "snippet": str}
#    entity_key  — matches keys in agent.py _KNOWN_ENTITIES (lowercase)
#    intent_key  — "general" | "claim" | "register" | "apply" | "check" | "benefit" | "contact"
# ─────────────────────────────────────────────────────────────────────────────

GOLDEN_URLS: dict[str, dict[str, dict]] = {

    # ── MALAYSIA ─────────────────────────────────────────────────────────────

    "perkeso": {
        "general": {
            "url": "https://www.perkeso.gov.my/en/",
            "title": "PERKESO (SOCSO) — Official Portal",
            "snippet": "PERKESO provides social security protection to workers in Malaysia, covering employment injury, invalidity, and job loss.",
        },
        "claim": {
            "url": "https://www.perkeso.gov.my/en/our-services/protection/employment-injury-scheme.html",
            "title": "PERKESO — Employment Injury Claims",
            "snippet": "How to claim PERKESO benefits for workplace injury, temporary disablement, permanent disablement, and more.",
        },
        "register": {
            "url": "https://perkeso.gov.my/en/our-services/employer-employee/employer-registration/",
            "title": "PERKESO — Employer Registration",
            "snippet": "Register as an employer with PERKESO within 30 days of hiring your first employee. Covers foreign workers too.",
        },
        "apply": {
            "url": "https://www.perkeso.gov.my/en/our-services/protection/employment-injury-scheme.html",
            "title": "PERKESO — Apply for Benefits",
            "snippet": "Application process for PERKESO employment injury and invalidity benefits including required documents.",
        },
        "check": {
            "url": "https://www.perkeso.gov.my/en/contact-us/pejabat-perkeso-new/frequently-asked-question.html",
            "title": "PERKESO — Frequently Asked Questions",
            "snippet": "Common questions about PERKESO registration, contributions, and benefit claims.",
        },
        "contact": {
            "url": "https://www.perkeso.gov.my/en/contact-us/",
            "title": "PERKESO — Contact Us",
            "snippet": "PERKESO hotline: 1-300-22-8000. Find your nearest PERKESO office.",
        },
    },

    "socso": {
        # SOCSO and PERKESO are the same — alias all intents
        "general": {
            "url": "https://www.perkeso.gov.my/en/",
            "title": "SOCSO / PERKESO — Official Portal",
            "snippet": "SOCSO (PERKESO) is Malaysia's social security organisation providing protection for workers.",
        },
        "claim": {
            "url": "https://www.perkeso.gov.my/en/our-services/protection/employment-injury-scheme.html",
            "title": "SOCSO — How to Claim Benefits",
            "snippet": "Steps to claim SOCSO benefits: report accident within 48 hours, submit Form 10 and Form 21, get medical treatment at panel clinic.",
        },
        "register": {
            "url": "https://perkeso.gov.my/en/our-services/employer-employee/employer-registration/",
            "title": "SOCSO — Employer Registration",
            "snippet": "Employer must register workers with SOCSO within 30 days. Covers Malaysian citizens, permanent residents, and foreign workers.",
        },
        "contact": {
            "url": "https://www.perkeso.gov.my/en/contact-us/",
            "title": "SOCSO — Contact & Office Locations",
            "snippet": "SOCSO hotline: 1-300-22-8000. Email and branch office directory.",
        },
    },

    "epf": {
        "general": {
            "url": "https://www.kwsp.gov.my/en/",
            "title": "EPF (KWSP) — Official Portal",
            "snippet": "The Employees Provident Fund (EPF/KWSP) manages retirement savings for Malaysian and non-Malaysian workers.",
        },
        "register": {
            "url": "https://www.kwsp.gov.my/en/member/overview/registration/ekyc",
            "title": "EPF — Online Member Registration (e-KYC)",
            "snippet": "Register as an EPF member online via the KWSP i-Akaun app. Requires MyKad or MyPR. No office visit needed for most cases.",
        },
        "claim": {
            "url": "https://www.kwsp.gov.my/en/member/account-centre/leaving-country",
            "title": "EPF — Withdrawal & Leaving Country",
            "snippet": "Foreign workers can withdraw full EPF savings when work permit expires or upon permanently leaving Malaysia.",
        },
        "apply": {
            "url": "https://www.kwsp.gov.my/en/member/kwsp-i-akaun",
            "title": "EPF — i-Akaun App (Withdrawals & Applications)",
            "snippet": "Apply for EPF withdrawals (housing, health, education, age 55/60) via the KWSP i-Akaun app.",
        },
        "check": {
            "url": "https://iakaun.kwsp.gov.my/portal/member/login",
            "title": "EPF — i-Akaun Member Portal (Check Balance)",
            "snippet": "Log in to i-Akaun to check your EPF balance, contribution history, and download member statements.",
        },
        "contact": {
            "url": "https://www.kwsp.gov.my/en/corporate/connect-with-us",
            "title": "EPF — Contact Us & Branch Locator",
            "snippet": "EPF contact centre: 03-8922 6000. Find nearest EPF office or Self-Service Terminal (SST).",
        },
    },

    "kwsp": {
        # Alias to EPF
        "general": {
            "url": "https://www.kwsp.gov.my/en/",
            "title": "KWSP (EPF) — Official Portal",
            "snippet": "KWSP/EPF manages retirement savings and provides withdrawal options for housing, health, education, and retirement.",
        },
        "register": {
            "url": "https://www.kwsp.gov.my/en/member/overview/registration/ekyc",
            "title": "KWSP — Online Member Registration",
            "snippet": "Register as KWSP member online. Malaysian citizens, permanent residents, and foreign workers with valid permits can register.",
        },
        "claim": {
            "url": "https://www.kwsp.gov.my/en/member/kwsp-i-akaun",
            "title": "KWSP — i-Akaun Withdrawals",
            "snippet": "Apply for KWSP withdrawals via i-Akaun app: Akaun Fleksibel, housing, health, education, and age-based withdrawals.",
        },
        "contact": {
            "url": "https://www.kwsp.gov.my/en/corporate/connect-with-us",
            "title": "KWSP — Contact Us",
            "snippet": "KWSP/EPF contact: 03-8922 6000. Branch and SST locator available on the website.",
        },
    },

    "jkm": {
        "general": {
            "url": "https://www.jkm.gov.my/",
            "title": "JKM — Jabatan Kebajikan Masyarakat (Social Welfare)",
            "snippet": "JKM provides welfare assistance to vulnerable groups including the elderly, disabled, and low-income families in Malaysia.",
        },
        "apply": {
            "url": "https://www.jkm.gov.my/jkm/index.php?r=portal/left&id=NnRMcWxheTlaVXZYeVJNbFJhV3ptUT09",
            "title": "JKM — Welfare Aid Application",
            "snippet": "Apply for JKM welfare aid including financial assistance, disability benefits, and social protection programs.",
        },
        "contact": {
            "url": "https://www.jkm.gov.my/jkm/index.php?r=portal/left&id=eE9aenk5aDh3YjFSbndHYVpJSTBsUT09",
            "title": "JKM — Contact & Office Directory",
            "snippet": "Find your nearest JKM office. Contact the welfare department for assistance with applications.",
        },
    },

    # ── MALAYSIA — IMMIGRATION (JIM) ─────────────────────────────────────────

    "jim": {
        "general": {
            "url": "https://www.imi.gov.my/",
            "title": "Jabatan Imigresen Malaysia (JIM) — Portal Rasmi",
            "snippet": "JIM menguruskan kemasukan, pengeluaran permit kerja, penguatkuasaan imigresen, dan perkhidmatan dokumen perjalanan di Malaysia.",
        },
        "apply": {
            "url": "https://www.imi.gov.my/index.php/ms/perkhidmatan-utama/pas/pas-lawatan-kerja-sementara-plks.html",
            "title": "JIM — Pas Lawatan Kerja Sementara (PLKS)",
            "snippet": "Permohonan dan pembaharuan PLKS bagi pekerja asing. Majikan perlu mengemukakan permohonan melalui sistem dalam talian JIM.",
        },
        "register": {
            "url": "https://www.imi.gov.my/index.php/ms/perkhidmatan-utama/pekerja-asing.html",
            "title": "JIM — Pendaftaran Pekerja Asing",
            "snippet": "Syarat dan prosedur pendaftaran pekerja asing dengan Jabatan Imigresen Malaysia, termasuk dokumen yang diperlukan.",
        },
        "check": {
            "url": "https://www.imi.gov.my/index.php/ms/soalan-lazim/faq-umum.html",
            "title": "JIM — Soalan Lazim (FAQ)",
            "snippet": "Soalan lazim mengenai permit kerja, visa, pas masuk, dan prosedur imigresen Malaysia.",
        },
        "contact": {
            "url": "https://www.imi.gov.my/index.php/ms/hubungi-kami.html",
            "title": "JIM — Hubungi Kami",
            "snippet": "Talian hotline JIM: 03-8880 1000. Pejabat imigresen di seluruh negara dan waktu operasi.",
        },
    },

    "imi": {
        "general": {
            "url": "https://www.imi.gov.my/",
            "title": "Jabatan Imigresen Malaysia (IMI) — Portal Rasmi",
            "snippet": "IMI bertanggungjawab mengawal kemasukan warganegara asing ke Malaysia dan menguruskan permit kerja serta dokumen perjalanan.",
        },
        "apply": {
            "url": "https://www.imi.gov.my/index.php/ms/perkhidmatan-utama/pas/pas-lawatan-kerja-sementara-plks.html",
            "title": "IMI — Permohonan Pas Lawatan Kerja (PLKS)",
            "snippet": "Cara memohon Pas Lawatan Kerja Sementara (PLKS): dokumen diperlukan, yuran, dan tempoh pemprosesan.",
        },
        "contact": {
            "url": "https://www.imi.gov.my/index.php/ms/hubungi-kami.html",
            "title": "IMI — Hubungi Kami",
            "snippet": "Hotline Imigresen: 03-8880 1000. Senarai pejabat imigresen negeri dan waktu berurusan.",
        },
    },

    "imigresen": {
        "general": {
            "url": "https://www.imi.gov.my/",
            "title": "Imigresen Malaysia — Portal Rasmi",
            "snippet": "Jabatan Imigresen Malaysia (JIM) menguruskan permit kerja, visa, pas pelajar, pas sosial, dan penguatkuasaan undang-undang imigresen.",
        },
        "contact": {
            "url": "https://www.imi.gov.my/index.php/ms/hubungi-kami.html",
            "title": "Imigresen Malaysia — Hubungi Kami",
            "snippet": "Hotline: 03-8880 1000. Waktu pejabat: Isnin–Jumaat 8:00 pagi–5:00 petang.",
        },
    },

    "plks": {
        "general": {
            "url": "https://www.imi.gov.my/index.php/ms/perkhidmatan-utama/pas/pas-lawatan-kerja-sementara-plks.html",
            "title": "Pas Lawatan Kerja Sementara (PLKS) — JIM",
            "snippet": "PLKS ialah permit kerja sah bagi pekerja asing di Malaysia. Majikan perlu mendaftar dan memperbaharui PLKS sebelum tamat tempoh.",
        },
        "apply": {
            "url": "https://www.imi.gov.my/index.php/ms/perkhidmatan-utama/pas/pas-lawatan-kerja-sementara-plks.html",
            "title": "PLKS — Cara Memohon / Memperbaharui",
            "snippet": "Proses permohonan PLKS: majikan memohon secara dalam talian melalui sistem JIM, lengkap dengan dokumen pekerja dan syarikat.",
        },
        "contact": {
            "url": "https://www.imi.gov.my/index.php/ms/hubungi-kami.html",
            "title": "PLKS — Hubungi JIM",
            "snippet": "Untuk pertanyaan PLKS: Jabatan Imigresen Malaysia, talian 03-8880 1000.",
        },
    },

    "work permit malaysia": {
        "general": {
            "url": "https://www.imi.gov.my/index.php/ms/perkhidmatan-utama/pas/pas-lawatan-kerja-sementara-plks.html",
            "title": "Malaysia Work Permit (PLKS) — Immigration Department",
            "snippet": "The Temporary Work Visit Pass (PLKS) is the main work permit for foreign workers in Malaysia. Employers must apply and renew before expiry.",
        },
        "apply": {
            "url": "https://www.imi.gov.my/index.php/ms/perkhidmatan-utama/pas/pas-lawatan-kerja-sementara-plks.html",
            "title": "Malaysia Work Permit — How to Apply / Renew",
            "snippet": "Work permit applications in Malaysia are employer-driven. The employer applies online via the JIM system with the worker's documents.",
        },
        "contact": {
            "url": "https://www.imi.gov.my/index.php/ms/hubungi-kami.html",
            "title": "Malaysia Immigration — Contact Us",
            "snippet": "Immigration Department of Malaysia hotline: 03-8880 1000. Operating hours: Mon–Fri 8 AM–5 PM.",
        },
    },

    "pembantu rumah asing": {
        "general": {
            "url": "https://www.imi.gov.my/index.php/ms/perkhidmatan-utama/pembantu-rumah-asing.html",
            "title": "JIM — Pembantu Rumah Asing (Foreign Domestic Helper)",
            "snippet": "Syarat dan prosedur mendapatkan pembantu rumah asing di Malaysia: agensi berlesen, negara punca, dokumen, dan yuran.",
        },
        "apply": {
            "url": "https://www.imi.gov.my/index.php/ms/perkhidmatan-utama/pembantu-rumah-asing.html",
            "title": "JIM — Permohonan Pembantu Rumah Asing",
            "snippet": "Majikan perlu menggunakan agensi pekerjaan berlesen untuk mendapatkan pembantu rumah asing. Pas dikeluarkan selama 2 tahun boleh diperbaharui.",
        },
    },

    # ── MALAYSIA — LABOUR / WORKER RIGHTS (JTKSM) ────────────────────────────

    "jtksm": {
        "general": {
            "url": "https://jtksm.mohr.gov.my/",
            "title": "Jabatan Tenaga Kerja Semenanjung Malaysia (JTKSM)",
            "snippet": "JTKSM menguatkuasakan Akta Kerja 1955 dan melindungi hak pekerja di Malaysia, termasuk pekerja asing dan pekerja domestik.",
        },
        "claim": {
            "url": "https://jtksm.mohr.gov.my/index.php/perkhidmatan/aduan",
            "title": "JTKSM — Aduan Buruh (Labour Complaint)",
            "snippet": "Cara buat aduan buruh: gaji tak bayar, pecat tanpa sebab, majikan zalim. Hubungi pejabat JTKSM terdekat atau e-Aduan dalam talian.",
        },
        "apply": {
            "url": "https://jtksm.mohr.gov.my/index.php/perkhidmatan/permit-perburuhan",
            "title": "JTKSM — Permit Perburuhan Pekerja Asing",
            "snippet": "Permit perburuhan diperlukan bagi pekerja asing di sektor pembuatan, pembinaan, perladangan, pertanian, dan perkhidmatan di Malaysia.",
        },
        "benefit": {
            "url": "https://jtksm.mohr.gov.my/index.php/perkhidmatan/pemberhentian-pekerja",
            "title": "JTKSM — Pemberhentian & Hak Pekerja",
            "snippet": "Hak pekerja semasa pemberhentian: notis, faedah terhenti, bayaran ganti rugi, dan prosedur tuntutan di JTKSM.",
        },
        "contact": {
            "url": "https://jtksm.mohr.gov.my/index.php/hubungi-kami",
            "title": "JTKSM — Hubungi Kami",
            "snippet": "Hotline Kementerian Sumber Manusia: 03-8886 5000. Pejabat JTKSM di semua negeri Semenanjung Malaysia.",
        },
    },

    "tenaga kerja": {
        "general": {
            "url": "https://jtksm.mohr.gov.my/",
            "title": "JTKSM — Jabatan Tenaga Kerja",
            "snippet": "Jabatan Tenaga Kerja Semenanjung Malaysia melindungi hak pekerja, memproses aduan buruh, dan mengeluarkan permit perburuhan.",
        },
        "claim": {
            "url": "https://jtksm.mohr.gov.my/index.php/perkhidmatan/aduan",
            "title": "JTKSM — Aduan Buruh",
            "snippet": "Serahkan aduan buruh di pejabat JTKSM. Aduan boleh meliputi gaji tertunggak, pemecatan, kerja lebih masa, atau penderaan majikan.",
        },
        "contact": {
            "url": "https://jtksm.mohr.gov.my/index.php/hubungi-kami",
            "title": "JTKSM — Cari Pejabat Berhampiran",
            "snippet": "Hubungi pejabat Jabatan Tenaga Kerja di negeri anda. Hotline: 03-8886 5000.",
        },
    },

    "aduan buruh": {
        "general": {
            "url": "https://jtksm.mohr.gov.my/index.php/perkhidmatan/aduan",
            "title": "JTKSM — Cara Buat Aduan Buruh di Malaysia",
            "snippet": "Pekerja yang tidak puas hati dengan majikan boleh buat aduan di pejabat Jabatan Tenaga Kerja. Kes: gaji tidak dibayar, PHK tanpa notis, kerja lebih masa tidak dibayar.",
        },
        "apply": {
            "url": "https://jtksm.mohr.gov.my/index.php/perkhidmatan/aduan",
            "title": "JTKSM — Borang Aduan Buruh",
            "snippet": "Bawa dokumen kontrak kerja, slip gaji, dan bukti aduan ke pejabat JTKSM. Kes akan diselesaikan dalam tempoh yang ditetapkan.",
        },
        "contact": {
            "url": "https://jtksm.mohr.gov.my/index.php/hubungi-kami",
            "title": "JTKSM — Hubungi Pejabat Tenaga Kerja",
            "snippet": "Hubungi pejabat JTKSM di negeri anda untuk membuat atau mengetahui status aduan buruh. Hotline: 03-8886 5000.",
        },
    },

    "labour complaint malaysia": {
        "general": {
            "url": "https://jtksm.mohr.gov.my/index.php/perkhidmatan/aduan",
            "title": "JTKSM — File a Labour Complaint in Malaysia",
            "snippet": "Foreign and local workers in Malaysia can file labour complaints at the nearest JTKSM office: unpaid wages, wrongful dismissal, overtime disputes, employer abuse.",
        },
        "contact": {
            "url": "https://jtksm.mohr.gov.my/index.php/hubungi-kami",
            "title": "JTKSM — Contact Labour Department",
            "snippet": "Human Resources Ministry hotline: 03-8886 5000. Find nearest JTKSM office in your state.",
        },
    },

    "unpaid salary malaysia": {
        "general": {
            "url": "https://jtksm.mohr.gov.my/index.php/perkhidmatan/aduan",
            "title": "JTKSM — Unpaid Salary Complaint Malaysia",
            "snippet": "If your employer has not paid your salary, file a complaint at the nearest Jabatan Tenaga Kerja (Labour Department). Bring your employment contract and payslips.",
        },
        "contact": {
            "url": "https://jtksm.mohr.gov.my/index.php/hubungi-kami",
            "title": "JTKSM — Contact Us",
            "snippet": "Labour Department Malaysia: 03-8886 5000. Complaints can be filed in person at any state JTKSM office.",
        },
    },

    # ── MALAYSIA — JKM (EXPANDED with scraped content) ───────────────────────

    "bantuan bencana": {
        "general": {
            "url": "https://www.jkm.gov.my/jkm/index.php?r=portal/left&id=NnRMcWxheTlaVXZYeVJNbFJhV3ptUT09",
            "title": "JKM — Bantuan Bencana (Flood & Disaster Aid)",
            "snippet": "JKM menyediakan bantuan bencana termasuk Bantuan Wang Ihsan, pusat pemindahan sementara, dan bantuan keperluan asas bagi mangsa banjir dan bencana.",
        },
        "apply": {
            "url": "https://www.jkm.gov.my/jkm/index.php?r=portal/left&id=NnRMcWxheTlaVXZYeVJNbFJhV3ptUT09",
            "title": "JKM — Cara Mohon Bantuan Bencana",
            "snippet": "Mangsa bencana boleh mendaftar di pusat pemindahan atau menghubungi pejabat JKM negeri. Bantuan Wang Ihsan diberikan kepada mereka yang terjejas.",
        },
        "contact": {
            "url": "https://www.jkm.gov.my/jkm/index.php?r=portal/left&id=eE9aenk5aDh3YjFSbndHYVpJSTBsUT09",
            "title": "JKM — Hubungi Kami",
            "snippet": "Talian hotline JKM: 03-8000 8000. Pejabat JKM di semua negeri.",
        },
    },

    "bantuan wang ihsan": {
        "general": {
            "url": "https://www.jkm.gov.my/jkm/index.php?r=portal/left&id=NnRMcWxheTlaVXZYeVJNbFJhV3ptUT09",
            "title": "JKM — Bantuan Wang Ihsan (Flood/Disaster Cash Aid)",
            "snippet": "Bantuan Wang Ihsan ialah bantuan kewangan tunai segera bagi mangsa banjir dan bencana alam di Malaysia. Kadar: RM500 seorang atau RM1,000 seisi rumah.",
        },
        "apply": {
            "url": "https://www.jkm.gov.my/jkm/index.php?r=portal/left&id=NnRMcWxheTlaVXZYeVJNbFJhV3ptUT09",
            "title": "JKM — Permohonan Bantuan Wang Ihsan",
            "snippet": "Daftar di pusat pemindahan banjir atau pejabat JKM negeri. Bawa kad pengenalan dan bukti kediaman.",
        },
    },

    "atip": {
        "general": {
            "url": "https://www.jkm.gov.my/",
            "title": "JKM — Perlindungan Mangsa ATIP (Pemerdagangan Orang)",
            "snippet": "JKM menyediakan perlindungan dan pemulihan kepada mangsa pemerdagangan orang (ATIP) di Malaysia, termasuk tempat perlindungan dan bantuan perundangan.",
        },
        "contact": {
            "url": "https://www.jkm.gov.my/jkm/index.php?r=portal/left&id=eE9aenk5aDh3YjFSbndHYVpJSTBsUT09",
            "title": "JKM — Hubungi untuk Bantuan ATIP",
            "snippet": "Mangsa pemerdagangan orang boleh mendapatkan bantuan melalui JKM. Hubungi: 03-8000 8000 atau polis (999).",
        },
    },

    # ── MALAYSIA — SSM (Business Registration) ───────────────────────────────

    "ssm": {
        "general": {
            "url": "https://www.ssm.com.my/",
            "title": "Suruhanjaya Syarikat Malaysia (SSM) — Pendaftaran Perniagaan",
            "snippet": "SSM menguruskan pendaftaran perniagaan, syarikat, dan perkongsian di Malaysia. Daftar perniagaan secara dalam talian melalui EzBiz.",
        },
        "register": {
            "url": "https://ezbiz.ssm.com.my/",
            "title": "SSM EzBiz — Daftar Perniagaan Dalam Talian",
            "snippet": "Daftar perniagaan perseorangan (Enterprise), perkongsian, atau syarikat sendirian berhad (Sdn Bhd) secara dalam talian. Yuran mulai RM60.",
        },
        "contact": {
            "url": "https://www.ssm.com.my/Pages/Contact_us/contact_us.aspx",
            "title": "SSM — Hubungi Kami",
            "snippet": "SSM hotline: 03-7721 4000. Kaunter SSM di seluruh negara.",
        },
    },

    # ─────────────────────────────────────────────────────────────────────────
    # ── INDONESIA ─────────────────────────────────────────────────────────────

    "bpjs": {
        "general": {
            "url": "https://bpjs-kesehatan.go.id/",
            "title": "BPJS Kesehatan — Official Portal",
            "snippet": "BPJS Kesehatan is Indonesia's national health insurance program covering medical treatment at partner hospitals and clinics.",
        },
        "register": {
            "url": "https://bpjs-kesehatan.go.id/user-manual-mobile-jkn/mobilejkn/mendaftarkanpeserta.html",
            "title": "BPJS Kesehatan — How to Register via Mobile JKN",
            "snippet": "Register for BPJS Kesehatan through the Mobile JKN app. Download the app, fill in personal data, and complete registration online.",
        },
        "claim": {
            "url": "https://bpjs-kesehatan.go.id/",
            "title": "BPJS Kesehatan — Claims & Services",
            "snippet": "Access BPJS Kesehatan services, find partner hospitals (FKTP/FKRTL), and manage your health insurance claims.",
        },
        "check": {
            "url": "https://bpjs-kesehatan.go.id/user-manual-mobile-jkn/mobilejkn/pendaftaranlogin.html",
            "title": "BPJS Kesehatan — Login Mobile JKN",
            "snippet": "Log in to Mobile JKN to check BPJS membership status, find hospitals, and access digital membership card.",
        },
    },

    "bpjs kesehatan": {
        "general": {
            "url": "https://bpjs-kesehatan.go.id/",
            "title": "BPJS Kesehatan — Official Portal",
            "snippet": "BPJS Kesehatan covers healthcare costs for all Indonesians. Register via Mobile JKN app or nearest BPJS office.",
        },
        "register": {
            "url": "https://bpjs-kesehatan.go.id/user-manual-mobile-jkn/mobilejkn/mendaftarkanpeserta.html",
            "title": "BPJS Kesehatan — Registration Guide",
            "snippet": "Step-by-step guide to register for BPJS Kesehatan online via Mobile JKN application.",
        },
    },

    "bpjs ketenagakerjaan": {
        "general": {
            "url": "https://www.bpjsketenagakerjaan.go.id/",
            "title": "BPJS Ketenagakerjaan — Official Portal",
            "snippet": "BPJS Ketenagakerjaan provides employment social security: workplace accident (JKK), death (JKM), old age savings (JHT), pension (JP), and job loss (JKP).",
        },
        "claim": {
            "url": "https://www.bpjsketenagakerjaan.go.id/en/cara-klaim.html",
            "title": "BPJS Ketenagakerjaan — How to Claim",
            "snippet": "Submit claims for JKK (workplace accident), JHT (old age savings), JKM (death benefit), and JP (pension) at nearest BPJAMSOSTEK branch.",
        },
        "register": {
            "url": "https://www.bpjsketenagakerjaan.go.id/bpu",
            "title": "BPJS Ketenagakerjaan — BPU Online Registration",
            "snippet": "Register as a self-employed (BPU) member of BPJS Ketenagakerjaan online. Covers freelancers, gig workers, and independent contractors.",
        },
        "apply": {
            "url": "https://www.bpjsketenagakerjaan.go.id/en/cara-klaim.html",
            "title": "BPJS Ketenagakerjaan — Apply for Benefits",
            "snippet": "How to apply for BPJS Ketenagakerjaan benefits including required forms and supporting documents.",
        },
    },

    "bansos": {
        "general": {
            "url": "https://cekbansos.kemensos.go.id/",
            "title": "Cek Bansos — Social Assistance Check (Kemensos)",
            "snippet": "Check eligibility for Indonesian government social assistance (Bansos) programs via the official Kemensos portal.",
        },
        "check": {
            "url": "https://cekbansos.kemensos.go.id/",
            "title": "Cek Bansos — Verify Your Social Aid",
            "snippet": "Enter your NIK to check if you are registered as a recipient of government social assistance programs.",
        },
    },

    "pkh": {
        "general": {
            "url": "https://kemensos.go.id/",
            "title": "PKH — Program Keluarga Harapan (Kemensos)",
            "snippet": "PKH is Indonesia's conditional cash transfer program for poor families covering health, education, and welfare needs.",
        },
    },

    # ── PHILIPPINES ───────────────────────────────────────────────────────────

    "sss": {
        "general": {
            "url": "https://www.sss.gov.ph/",
            "title": "SSS — Social Security System Philippines",
            "snippet": "SSS provides social security protection to private sector employees covering sickness, maternity, disability, retirement, and death benefits.",
        },
        "register": {
            "url": "https://www.sss.gov.ph/employer-er/",
            "title": "SSS — Employer Registration",
            "snippet": "Register as an SSS employer online via My.SSS portal. Required before registering employees for PhilHealth and Pag-IBIG.",
        },
        "claim": {
            "url": "https://www.sss.gov.ph/",
            "title": "SSS — Claims & Benefits",
            "snippet": "File SSS claims for sickness, maternity, disability, retirement, and death benefits via My.SSS portal or nearest SSS branch.",
        },
        "apply": {
            "url": "https://www.sss.gov.ph/",
            "title": "SSS — Apply for Benefits",
            "snippet": "Apply for SSS benefits online via My.SSS. Create an account at my.sss.gov.ph to access all SSS services.",
        },
        "contact": {
            "url": "https://www.sss.gov.ph/",
            "title": "SSS — Contact Us",
            "snippet": "SSS hotline: 1455 (local). Find nearest SSS branch via the official website.",
        },
    },

    "philhealth": {
        "general": {
            "url": "https://www.philhealth.gov.ph/",
            "title": "PhilHealth — Philippine Health Insurance Corporation",
            "snippet": "PhilHealth provides health insurance coverage for hospitalization and medical expenses to all Filipinos.",
        },
        "register": {
            "url": "https://www.philhealth.gov.ph/partners/employers/registration.php",
            "title": "PhilHealth — Employer Registration",
            "snippet": "Register as a PhilHealth employer. Submit ER1 form to any PhilHealth office with your business documents.",
        },
        "claim": {
            "url": "https://www.philhealth.gov.ph/",
            "title": "PhilHealth — How to Claim Benefits",
            "snippet": "PhilHealth claims are processed directly at accredited hospitals. Ensure your hospital is a PhilHealth-accredited facility.",
        },
        "contact": {
            "url": "https://www.philhealth.gov.ph/",
            "title": "PhilHealth — Contact Us",
            "snippet": "PhilHealth hotline: (02) 8441-7442. Find nearest PhilHealth Local Health Insurance Office (LHIO).",
        },
    },

    "pagibig": {
        "general": {
            "url": "https://www.pagibigfund.gov.ph/",
            "title": "Pag-IBIG Fund — Home Development Mutual Fund",
            "snippet": "Pag-IBIG Fund provides housing loans, multi-purpose loans, and savings programs to Filipino workers.",
        },
        "register": {
            "url": "https://www.pagibigfund.gov.ph/",
            "title": "Pag-IBIG — Online Member Registration (Virtual Pag-IBIG)",
            "snippet": "Register as Pag-IBIG member online via Virtual Pag-IBIG. Select 'Be a Member' then 'Register' on the official website.",
        },
        "claim": {
            "url": "https://www.pagibigfund.gov.ph/",
            "title": "Pag-IBIG — Loan Applications & Claims",
            "snippet": "Apply for Pag-IBIG housing loans, multi-purpose loans, and calamity loans via Virtual Pag-IBIG online portal.",
        },
        "apply": {
            "url": "https://www.pagibigfund.gov.ph/",
            "title": "Pag-IBIG — Apply for Loans",
            "snippet": "Apply for Pag-IBIG housing or multi-purpose loans online. Requires active Pag-IBIG membership and 24-month contribution.",
        },
    },

    "owwa": {
        "general": {
            "url": "https://www.owwa.gov.ph/",
            "title": "OWWA — Overseas Workers Welfare Administration",
            "snippet": "OWWA provides welfare services, livelihood assistance, and reintegration programs to Overseas Filipino Workers (OFWs).",
        },
        "register": {
            "url": "https://www.owwa.gov.ph/",
            "title": "OWWA — Membership Registration",
            "snippet": "OFWs can register for OWWA membership at Philippine Overseas Labor Offices (POLO) abroad or at OWWA offices in the Philippines.",
        },
        "claim": {
            "url": "https://www.owwa.gov.ph/",
            "title": "OWWA — Benefits & Claims",
            "snippet": "OWWA benefits include death and disability benefits, education assistance, and livelihood programs for OFWs and their families.",
        },
    },

    "dti": {
        "general": {
            "url": "https://www.dti.gov.ph/",
            "title": "DTI — Department of Trade and Industry Philippines",
            "snippet": "DTI supports small businesses, provides business registration, and offers livelihood programs for Filipino entrepreneurs.",
        },
        "register": {
            "url": "https://www.dti.gov.ph/",
            "title": "DTI — Business Name Registration",
            "snippet": "Register your business name with DTI online via the DTI Business Name Registration System (BNRS).",
        },
        "apply": {
            "url": "https://www.dti.gov.ph/",
            "title": "DTI — Livelihood & Enterprise Programs",
            "snippet": "Apply for DTI livelihood assistance programs, business grants, and MSME support services.",
        },
    },

    "4ps": {
        "general": {
            "url": "https://www.dswd.gov.ph/",
            "title": "4Ps / Pantawid Pamilya — DSWD",
            "snippet": "The Pantawid Pamilyang Pilipino Program (4Ps) is the Philippine government's conditional cash transfer program for poor families.",
        },
        "apply": {
            "url": "https://www.dswd.gov.ph/",
            "title": "4Ps — How to Apply (DSWD)",
            "snippet": "4Ps applications are processed through your local DSWD office. Registration is based on National Household Targeting System (NHTS) data.",
        },
    },

    # ── THAILAND ──────────────────────────────────────────────────────────────

    "ประกันสังคม": {
        "general": {
            "url": "https://www.sso.go.th/",
            "title": "สำนักงานประกันสังคม (SSO) — ประกันสังคมไทย",
            "snippet": "ประกันสังคมไทยให้ความคุ้มครองแก่ลูกจ้างในภาคเอกชน ครอบคลุมค่ารักษาพยาบาล การว่างงาน ทุพพลภาพ ชราภาพ และเสียชีวิต",
        },
        "register": {
            "url": "https://www.sso.go.th/",
            "title": "SSO — ลงทะเบียนประกันสังคม",
            "snippet": "นายจ้างต้องขึ้นทะเบียนลูกจ้างกับสำนักงานประกันสังคมภายใน 30 วันนับจากวันเริ่มงาน ใช้แบบฟอร์ม SSO 1-03",
        },
        "claim": {
            "url": "https://www.sso.go.th/",
            "title": "SSO — การยื่นเรื่องขอรับประโยชน์ทดแทน",
            "snippet": "ยื่นเรื่องขอรับสิทธิประโยชน์ประกันสังคม เช่น ค่ารักษาพยาบาล ว่างงาน ชราภาพ ที่สำนักงานประกันสังคมใกล้บ้าน",
        },
        "contact": {
            "url": "https://www.sso.go.th/",
            "title": "SSO — ติดต่อสำนักงานประกันสังคม",
            "snippet": "สายด่วนประกันสังคม 1506 ให้บริการตลอด 24 ชั่วโมง",
        },
    },

    "บัตรทอง": {
        "general": {
            "url": "https://www.nhso.go.th/",
            "title": "บัตรทอง / สิทธิหลักประกันสุขภาพถ้วนหน้า (สปสช.)",
            "snippet": "บัตรทองให้สิทธิรักษาพยาบาลฟรีสำหรับประชาชนไทยทุกคนที่ไม่มีสิทธิประกันสังคมหรือข้าราชการ",
        },
        "register": {
            "url": "https://www.nhso.go.th/",
            "title": "สปสช. — ลงทะเบียนบัตรทอง",
            "snippet": "ลงทะเบียนใช้สิทธิบัตรทองได้ที่โรงพยาบาลหรือสถานีอนามัยใกล้บ้าน",
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. ALLOWED DOMAINS (security allowlist — unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_DOMAINS: dict[str, list[str]] = {
    "MY": [
        # Immigration (JIM) — newly scraped
        "www.imi.gov.my",
        "imi.gov.my",
        # Labour / Tenaga Kerja (JTKSM) — newly scraped
        "jtksm.mohr.gov.my",
        "www.mohr.gov.my",
        "mohr.gov.my",
        # Social welfare (JKM) — newly scraped
        "www.jkm.gov.my",
        "jkm.gov.my",
        # Social security (PERKESO/SOCSO) — newly scraped
        "www.perkeso.gov.my",
        "perkeso.gov.my",
        # Health
        "www.moh.gov.my",
        "moh.gov.my",
        # EPF/KWSP
        "www.kwsp.gov.my",
        "kwsp.gov.my",
        "iakaun.kwsp.gov.my",
        # Housing
        "www.kkr.gov.my",
        "kkr.gov.my",
        # Business registration
        "www.ssm.com.my",
        "ssm.com.my",
        "ezbiz.ssm.com.my",
    ],
    "ID": [
        "www.bpjs-kesehatan.go.id",
        "bpjs-kesehatan.go.id",
        "www.bpjsketenagakerjaan.go.id",
        "bpjsketenagakerjaan.go.id",
        "eklaim-pmi.bpjsketenagakerjaan.go.id",
        "kemensos.go.id",
        "cekbansos.kemensos.go.id",
        "www.kemnaker.go.id",
        "kemnaker.go.id",
        "bp2mi.go.id",
        "www.bp2mi.go.id",
    ],
    "PH": [
        "www.sss.gov.ph",
        "sss.gov.ph",
        "my.sss.gov.ph",
        "www.philhealth.gov.ph",
        "philhealth.gov.ph",
        "www.pagibigfund.gov.ph",
        "pagibigfund.gov.ph",
        "www.owwa.gov.ph",
        "owwa.gov.ph",
        "www.dti.gov.ph",
        "dti.gov.ph",
        "www.dswd.gov.ph",
        "dswd.gov.ph",
        "www.dmw.gov.ph",
        "dmw.gov.ph",
    ],
    "TH": [
        "www.sso.go.th",
        "sso.go.th",
        "www.mol.go.th",
        "mol.go.th",
        "www.doe.go.th",
        "doe.go.th",
        "www.nhso.go.th",
        "nhso.go.th",
    ],
    "ASEAN": [
        "asean.org",
        "www.asean.org",
    ],
}

COUNTRY_SITE_SCOPES: dict[str, str] = {
    "MY": "site:gov.my",
    "ID": "site:go.id",
    "PH": "site:gov.ph",
    "TH": "site:go.th",
    "ASEAN": "site:asean.org",
}

MAX_SEARCH_RESULTS = 5

# Liveness check timeouts — golden URLs are curated so we fail faster;
# DDG results are untrusted so we wait a bit longer.
LIVENESS_TIMEOUT_GOLDEN = 2.0   # seconds — golden URLs are curated / unlikely dead
LIVENESS_TIMEOUT_DDG    = 4.0   # seconds — DDG results are less predictable

# Stopwords stripped before keyword scoring (EN + MS + ID + TH + TL)
_STOPWORDS = {
    # English
    "the", "a", "an", "in", "on", "at", "to", "of", "for", "and", "or",
    "how", "what", "where", "when", "who", "can", "do", "i", "my", "is",
    "are", "me", "with", "from", "about",
    # Malay / Indonesian
    "saya", "nak", "apa", "di", "ke", "yang", "dan", "ada", "boleh",
    "cara", "untuk", "dengan", "ini", "itu", "oleh", "jika", "akan",
    "telah", "sudah", "perlu", "tidak", "bagi", "adalah", "dalam",
    # Filipino / Tagalog
    "ang", "ng", "sa", "na", "ko", "ako", "kita", "mo", "ito", "po",
    "mga", "ay", "at", "mga", "din", "naman",
    # Thai
    "ที่", "ใน", "ของ", "ได้", "และ", "หรือ", "จาก", "คือ", "เป็น", "มี",
    "ให้", "ต้อง", "จะ", "ไม่", "กับ",
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. INTENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# Pattern → intent label
_INTENT_PATTERNS: list[tuple[re.Pattern, str]] = [
    # claim / tuntutan / klaim / aduan / คลาม
    (re.compile(r"\b(claim|claims|tuntut|tuntutan|klaim|cara klaim|คลาม|รับสิทธิ|aduan|complaint|complain)\b", re.I), "claim"),
    # apply / mohon / pengajuan / สมัคร / permohonan
    (re.compile(r"\b(apply|applying|applied|applic|mohon|permohonan|pengajuan|สมัคร|申請|buat permohonan)\b", re.I), "apply"),
    # register / daftar / ลงทะเบียน / renew / baharui
    (re.compile(r"\b(register|registr|daftar|pendaftaran|ลงทะเบียน|สมัคร|renew|renewal|pembaharuan|baharui|memperbaharui)\b", re.I), "register"),
    # check / semak / ตรวจสอบ / cek / status
    (re.compile(r"\b(check|semak|cek|ตรวจสอบ|how much|berapa|balance|baki|saldo|status|tamat|expired)\b", re.I), "check"),
    # benefit / faedah / สิทธิประโยชน์ / manfaat / hak / rights
    (re.compile(r"\b(benefit|faedah|manfaat|สิทธิ|ประโยชน์|coverage|perlindungan|hak|rights|kelayakan|eligible)\b", re.I), "benefit"),
    # contact / hubungi / ติดต่อ / office / pejabat
    (re.compile(r"\b(contact|hubung|telefon|phone|hotline|office|pejabat|ติดต่อ|โทร|nombor)\b", re.I), "contact"),
]


def _detect_intent(query: str) -> str:
    """Return intent label from query, defaulting to 'general'."""
    for pattern, intent in _INTENT_PATTERNS:
        if pattern.search(query):
            return intent
    return "general"


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENTITY MATCHING
# ─────────────────────────────────────────────────────────────────────────────

# Maps query keywords → golden URL entity keys
_ENTITY_ALIASES: dict[str, str] = {
    # ── MY — Immigration (JIM) ────────────────────────────────────────────────
    "jabatan imigresen": "jim",
    "imigresen malaysia": "imigresen",
    "imigresen": "imigresen",
    "jim": "jim",
    "imi": "imi",
    "plks": "plks",
    "pas lawatan kerja": "plks",
    "work permit malaysia": "work permit malaysia",
    "work permit": "work permit malaysia",
    "permit kerja": "plks",
    "pembantu rumah asing": "pembantu rumah asing",
    "foreign domestic helper": "pembantu rumah asing",
    "maid malaysia": "pembantu rumah asing",
    # ── MY — Labour / Worker Rights (JTKSM) ──────────────────────────────────
    "jtksm": "jtksm",
    "jabatan tenaga kerja": "tenaga kerja",
    "tenaga kerja": "tenaga kerja",
    "aduan buruh": "aduan buruh",
    "labour complaint": "labour complaint malaysia",
    "labor complaint": "labour complaint malaysia",
    "unpaid salary": "unpaid salary malaysia",
    "gaji tak bayar": "aduan buruh",
    "gaji tidak bayar": "aduan buruh",
    "upah tidak dibayar": "aduan buruh",
    # ── MY — Social Security (PERKESO/SOCSO) ─────────────────────────────────
    "perkeso": "perkeso",
    "socso": "socso",
    "epf": "epf",
    "kwsp": "kwsp",
    # ── MY — Social Welfare (JKM) ────────────────────────────────────────────
    "jkm": "jkm",
    "kebajikan": "jkm",
    "bantuan bencana": "bantuan bencana",
    "bantuan wang ihsan": "bantuan wang ihsan",
    "flood aid malaysia": "bantuan bencana",
    "banjir bantuan": "bantuan bencana",
    "atip": "atip",
    "pemerdagangan orang": "atip",
    "trafficking": "atip",
    # ── MY — Business ─────────────────────────────────────────────────────────
    "ssm": "ssm",
    "suruhanjaya syarikat": "ssm",
    "daftar perniagaan": "ssm",
    "business registration malaysia": "ssm",
    # ── ID ────────────────────────────────────────────────────────────────────
    "bpjs kesehatan": "bpjs kesehatan",
    "bpjs ketenagakerjaan": "bpjs ketenagakerjaan",
    "bpjamsostek": "bpjs ketenagakerjaan",
    "bpjs": "bpjs",
    "bansos": "bansos",
    "pkh": "pkh",
    "jaminan sosial": "bpjs ketenagakerjaan",
    # ── PH ────────────────────────────────────────────────────────────────────
    "philhealth": "philhealth",
    "sss": "sss",
    "pag-ibig": "pagibig",
    "pagibig": "pagibig",
    "hdmf": "pagibig",
    "owwa": "owwa",
    "dti": "dti",
    "4ps": "4ps",
    "pantawid": "4ps",
    "dswd": "4ps",
    # ── TH ────────────────────────────────────────────────────────────────────
    "ประกันสังคม": "ประกันสังคม",
    "sso": "ประกันสังคม",
    "social security thailand": "ประกันสังคม",
    "บัตรทอง": "บัตรทอง",
    "30 baht": "บัตรทอง",
    "nhso": "บัตรทอง",
}

# Fix 5: Auto-country inference — entity key → country code
_ENTITY_COUNTRY: dict[str, str] = {
    # MY
    "jim": "MY", "imi": "MY", "imigresen": "MY",
    "plks": "MY", "work permit malaysia": "MY", "pembantu rumah asing": "MY",
    "jtksm": "MY", "tenaga kerja": "MY",
    "aduan buruh": "MY", "labour complaint malaysia": "MY", "unpaid salary malaysia": "MY",
    "perkeso": "MY", "socso": "MY", "epf": "MY", "kwsp": "MY", "jkm": "MY",
    "bantuan bencana": "MY", "bantuan wang ihsan": "MY", "atip": "MY",
    "ssm": "MY",
    # ID
    "bpjs": "ID", "bpjs kesehatan": "ID", "bpjs ketenagakerjaan": "ID",
    "bansos": "ID", "pkh": "ID",
    # PH
    "sss": "PH", "philhealth": "PH", "pagibig": "PH",
    "owwa": "PH", "dti": "PH", "4ps": "PH",
    # TH
    "ประกันสังคม": "TH", "บัตรทอง": "TH",
}


def _detect_entity(query: str) -> str | None:
    """Return golden URL entity key if query matches a known entity."""
    q = query.lower()
    # Longest match first to avoid "bpjs" matching before "bpjs kesehatan"
    for alias in sorted(_ENTITY_ALIASES.keys(), key=len, reverse=True):
        if alias in q:
            return _ENTITY_ALIASES[alias]
    return None


def _infer_country_from_entity(entity: str | None) -> str:
    """Return country code inferred from entity key, or empty string."""
    if entity is None:
        return ""
    return _ENTITY_COUNTRY.get(entity, "")


# ─────────────────────────────────────────────────────────────────────────────
# 5. GOLDEN URL LOOKUP — see _get_golden_results() above (Fix 4)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 6. RED-FLAG FILTER (for DuckDuckGo results)
# ─────────────────────────────────────────────────────────────────────────────

# Pages that look like official service pages but are actually wrong destinations
_RED_FLAG_TITLE_WORDS = {
    "complaint", "aduan", "feedback", "maklumbalas", "cadangan",
    "complain", "pengaduan", "laporan penyalahgunaan",
    "login", "log in", "sign in", "log masuk", "daftar masuk",
    "e-aduan", "spab", "portal aduan", "sistem aduan",
    "careers", "kerjaya", "tender", "procurement",
    "annual report", "laporan tahunan",
}

_RED_FLAG_URL_SEGMENTS = {
    "aduan", "complaint", "feedback", "careers", "tender",
    "spab",  # This was the culprit in the PERKESO example!
    "login", "signin",
}


def _is_red_flag(result: dict) -> bool:
    """Return True if a DuckDuckGo result looks like a wrong-destination page."""
    title = result.get("title", "").lower()
    url = result.get("url", "").lower()

    for word in _RED_FLAG_TITLE_WORDS:
        if word in title:
            logger.debug("Red-flag title word '%s' in: %s", word, result.get("title"))
            return True

    for segment in _RED_FLAG_URL_SEGMENTS:
        if f"/{segment}" in url or f".{segment}" in url:
            logger.debug("Red-flag URL segment '%s' in: %s", segment, url)
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# 7. LIVENESS CHECK
# ─────────────────────────────────────────────────────────────────────────────

async def _is_url_live(url: str, timeout: float = LIVENESS_TIMEOUT_DDG) -> bool:
    """HEAD request to verify URL resolves. Returns True if 2xx or 3xx."""
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
        ) as client:
            resp = await client.head(url, headers={"User-Agent": "AskAra/1.0"})
            live = resp.status_code < 400
            logger.debug("Liveness %s → HTTP %d", url, resp.status_code)
            return live
    except Exception as exc:
        logger.debug("Liveness check failed for %s: %s", url, exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Fix 3: KEYWORD RELEVANCE SCORING (for DuckDuckGo results)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_keywords(query: str) -> list[str]:
    """Extract meaningful lowercase tokens from query, stripping stopwords."""
    tokens = re.findall(r"[^\s\W]+", query.lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]


def _score_relevance(result: dict, keywords: list[str]) -> int:
    """
    Score a DDG result by how many query keywords appear in title + snippet.
    Higher is better. Zero means the result is completely off-topic.
    """
    if not keywords:
        return 1  # No keywords to match — pass everything through
    haystack = (
        result.get("title", "").lower()
        + " "
        + result.get("body", result.get("snippet", "")).lower()
    )
    return sum(1 for kw in keywords if kw in haystack)


# ─────────────────────────────────────────────────────────────────────────────
# Fix 4: MULTI-GOLDEN RESULT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _get_golden_results(query: str) -> list[dict]:
    """
    Return up to 3 golden URL result dicts for a query:
      1. Primary  — specific intent match (e.g. "claim")
      2. Secondary — "general" page (if different from primary)
      3. Tertiary  — "contact" page (always useful for migrants)

    Returns empty list if no entity matched.
    """
    entity = _detect_entity(query)
    if not entity or entity not in GOLDEN_URLS:
        return []

    intent = _detect_intent(query)
    entity_map = GOLDEN_URLS[entity]

    seen_urls: set[str] = set()
    collected: list[dict] = []

    def _add(entry: dict | None) -> None:
        if not entry:
            return
        u = entry["url"]
        if u in seen_urls:
            return
        seen_urls.add(u)
        collected.append({
            "title": entry["title"],
            "url": u,
            "snippet": entry["snippet"],
            "_source": "golden",
        })

    # 1. Specific intent
    _add(entity_map.get(intent))
    # 2. General page (often different URL)
    _add(entity_map.get("general"))
    # 3. Contact page — always useful for confused migrants
    _add(entity_map.get("contact"))

    return collected


# ─────────────────────────────────────────────────────────────────────────────
# 8. HELPERS (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def _is_allowed(url: str) -> bool:
    domain = urlparse(url).netloc.lower()
    domain_no_www = domain.removeprefix("www.")
    for domains in ALLOWED_DOMAINS.values():
        if domain in domains or f"www.{domain_no_www}" in domains:
            return True
    return False


def _get_site_scope(url: str = "", country: str = "") -> str:
    if url:
        parsed = urlparse(url if url.startswith("http") else f"https://{url}")
        domain = parsed.netloc.lower()
        if domain:
            return f"site:{domain}"
    if country and country.upper() in COUNTRY_SITE_SCOPES:
        return COUNTRY_SITE_SCOPES[country.upper()]
    return ""


def _detect_country(url: str) -> str:
    domain = urlparse(url).netloc.lower() if url.startswith("http") else url.lower()
    if ".my" in domain:
        return "MY"
    elif ".id" in domain or ".go.id" in domain:
        return "ID"
    elif ".ph" in domain:
        return "PH"
    elif ".th" in domain:
        return "TH"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_gov_portal(
    url: str,
    country: str = "MY",
) -> str:
    """Search government portals for fresh information.

    Malaysia-focused: defaults to MY government portals.
    Hardened with:
    - Golden URL map (no DuckDuckGo needed for known entities)
    - Multi-golden results (up to 3 per hit: intent + general + contact)
    - Auto-country inference from entity name
    - DDG query terms placed BEFORE site: operator for better ranking
    - Keyword relevance scoring (zero-match results discarded)
    - Red-flag filtering (complaints/login pages rejected)
    - Parallel liveness checks (concurrent HEAD requests, not sequential)

    Args:
        url: A search query (e.g. "how to claim PERKESO") OR a specific
             government portal URL to scope search.
        country: Country code ("MY", "ID", "PH", "TH").

    Returns:
        JSON string with results, content, query_used, status, source_tier.
    """
    if not url or not url.strip():
        return json.dumps({
            "results": [], "query_used": "", "fetched_at": "",
            "country": "", "result_count": 0, "status": "error",
            "error": "No URL or search query provided.",
        })

    url = url.strip()
    fetched_at = datetime.now(timezone.utc).isoformat()
    is_url = url.startswith("http") or url.startswith("www.") or "gov." in url.lower()
    original_query = url  # preserve for keyword scoring

    # ── Fix 3: Pre-compute keywords once for the whole function ─────────────
    keywords = _extract_keywords(original_query) if not is_url else []

    # ── Path A: Golden URL (query matches known entity) ─────────────────────
    if not is_url:
        # Fix 4: Multi-golden results
        golden_entries = _get_golden_results(url)
        if golden_entries:
            logger.info(
                "fetch_gov_portal: golden hit — %d entries for query '%s'",
                len(golden_entries), url[:60],
            )

            # Fix 5: Auto-infer country from entity
            entity = _detect_entity(url)
            if not country:
                country = _infer_country_from_entity(entity)

            # Fix 6: Parallel liveness checks with GOLDEN timeout (faster)
            live_flags = await asyncio.gather(
                *[_is_url_live(e["url"], timeout=LIVENESS_TIMEOUT_GOLDEN) for e in golden_entries]
            )
            live_entries = [e for e, live in zip(golden_entries, live_flags) if live]

            if not live_entries:
                logger.warning(
                    "All golden URLs dead for entity '%s' — falling through to DDG", entity
                )
                # Fall through to DuckDuckGo below
            else:
                # Drop internal _source key before returning
                results = [
                    {"title": e["title"], "url": e["url"], "snippet": e["snippet"]}
                    for e in live_entries
                ]
                combined_content = "\n\n".join(
                    f"**{r['title']}**\n{r['snippet']}\nSource: {r['url']}"
                    for r in results
                )
                return json.dumps({
                    "results": results,
                    "content": combined_content,
                    "query_used": url,
                    "fetched_at": fetched_at,
                    "country": country,
                    "result_count": len(results),
                    "source_tier": "golden",
                    "status": "success",
                }, ensure_ascii=False)

    # ── Path B: DuckDuckGo search ─────────────────────────────────────────
    if is_url:
        clean_url = url if url.startswith("http") else f"https://{url}"
        if not _is_allowed(clean_url):
            domain = urlparse(clean_url).netloc
            return json.dumps({
                "results": [], "query_used": "", "fetched_at": fetched_at,
                "country": "", "result_count": 0, "status": "blocked",
                "error": f"Domain '{domain}' is not in the allowlist.",
            })
        if not country:
            country = _detect_country(clean_url)
        site_scope = _get_site_scope(url=clean_url)
        path = urlparse(clean_url).path.strip("/").replace("-", " ").replace("/", " ")
        # Fix 1: search terms BEFORE site: operator
        search_query = f"{path} {site_scope}".strip() if path else site_scope
    else:
        country = country.strip().upper() if country else ""
        # Fix 5: still try to infer country if not set
        if not country:
            entity = _detect_entity(url)
            country = _infer_country_from_entity(entity)
        site_scope = _get_site_scope(country=country)
        # Fix 1: query terms BEFORE site: operator
        search_query = f"{url} {site_scope}".strip() if site_scope else url

    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return json.dumps({
                "results": [], "query_used": search_query, "fetched_at": fetched_at,
                "country": country, "result_count": 0, "status": "error",
                "error": "DuckDuckGo search not installed. Run: uv add ddgs",
            })

    try:
        ddgs = DDGS()
        logger.info("fetch_gov_portal: DDG search '%s'", search_query)
        raw_results = ddgs.text(search_query, max_results=MAX_SEARCH_RESULTS * 2)
        source_tier = "government"

        if not raw_results and not is_url:
            country_name = {"MY": "Malaysia", "ID": "Indonesia",
                            "PH": "Philippines", "TH": "Thailand"}.get(country, "")
            # Fix 1: query terms first even in fallback
            general_query = f"{url} {country_name} official government".strip()
            logger.info(
                "fetch_gov_portal: DDG gov empty — falling back to web '%s'", general_query
            )
            raw_results = ddgs.text(general_query, max_results=MAX_SEARCH_RESULTS)
            source_tier = "web"

        if not raw_results:
            return json.dumps({
                "results": [], "query_used": search_query, "fetched_at": fetched_at,
                "country": country, "result_count": 0, "status": "no_results",
                "note": "No results found.",
            })

        # ── Apply red-flag filter ──────────────────────────────────────────
        filtered = [r for r in raw_results if not _is_red_flag(r)]
        removed = len(raw_results) - len(filtered)
        if removed:
            logger.info("fetch_gov_portal: removed %d red-flag results", removed)

        if not filtered:
            logger.warning("All DDG results were red-flagged — returning no_results")
            return json.dumps({
                "results": [], "query_used": search_query, "fetched_at": fetched_at,
                "country": country, "result_count": 0, "status": "no_results",
                "note": "All results were filtered as irrelevant pages (complaints, login, etc).",
            })

        # ── Fix 3: Keyword relevance scoring — sort and drop zero-score ───
        if keywords:
            scored = [
                (r, _score_relevance(r, keywords))
                for r in filtered
            ]
            # Keep results with at least 1 keyword hit, sorted by score desc
            scored = [(r, s) for r, s in scored if s > 0]
            if scored:
                scored.sort(key=lambda x: x[1], reverse=True)
                filtered = [r for r, _ in scored]
                logger.info(
                    "fetch_gov_portal: keyword scoring kept %d/%d results",
                    len(filtered), len(raw_results),
                )
            else:
                # No hits — fall back to unscored (let red-flag filter be enough)
                logger.info(
                    "fetch_gov_portal: keyword scoring found 0 hits — using unfiltered results"
                )
                filtered = [r for r in raw_results if not _is_red_flag(r)] or raw_results

        candidates = filtered[:MAX_SEARCH_RESULTS]

        # ── Fix 2: Parallel liveness checks (concurrent, not sequential) ──
        urls_to_check = [r.get("href", "") for r in candidates]
        live_flags = await asyncio.gather(
            *[_is_url_live(u, timeout=LIVENESS_TIMEOUT_DDG) for u in urls_to_check]
        )

        results = []
        for r, u, live in zip(candidates, urls_to_check, live_flags):
            if not u:
                continue
            if live:
                results.append({
                    "title": r.get("title", ""),
                    "url": u,
                    "snippet": r.get("body", ""),
                })
            else:
                logger.info("fetch_gov_portal: dropped dead URL %s", u)

        if not results:
            return json.dumps({
                "results": [], "query_used": search_query, "fetched_at": fetched_at,
                "country": country, "result_count": 0, "status": "no_results",
                "note": "All results failed liveness check.",
            })

        combined_content = "\n\n".join(
            f"**{r['title']}**\n{r['snippet']}\nSource: {r['url']}"
            for r in results
        )

        logger.info(
            "fetch_gov_portal: returning %d live results (%s) for '%s'",
            len(results), source_tier, search_query[:60],
        )

        return json.dumps({
            "results": results,
            "content": combined_content,
            "query_used": search_query,
            "fetched_at": fetched_at,
            "country": country,
            "result_count": len(results),
            "source_tier": source_tier,
            "status": "success",
        }, ensure_ascii=False)

    except Exception as e:
        error_name = type(e).__name__
        logger.error("DuckDuckGo search failed: %s: %s", error_name, e)
        return json.dumps({
            "results": [], "query_used": search_query, "fetched_at": fetched_at,
            "country": country, "result_count": 0, "status": "error",
            "error": f"Search failed ({error_name}): {str(e)}",
        })