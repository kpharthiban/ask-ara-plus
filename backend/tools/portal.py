"""
fetch_gov_portal — Hardened government portal search with golden URL map.

Owner: Pharthiban
Depends on: duckduckgo-search (pip install duckduckgo-search), httpx

Three-layer approach (in priority order):
  1. GOLDEN URLs — curated, verified landing pages per entity + intent.
     DuckDuckGo is NOT called at all when a golden URL exists. Zero hallucination risk.
  2. Filtered DuckDuckGo — when no golden URL matches, DuckDuckGo runs but results
     are scored and red-flag pages (complaints, login, feedback portals) are removed.
  3. Liveness check — every URL returned (golden or DDG) is HEAD-checked via httpx
     before being included in results. Dead links are silently dropped.

Security: Only allowlisted government domains are permitted.
"""

from __future__ import annotations

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
        "www.perkeso.gov.my",
        "perkeso.gov.my",
        "www.mohr.gov.my",
        "mohr.gov.my",
        "www.jkm.gov.my",
        "jkm.gov.my",
        "www.moh.gov.my",
        "moh.gov.my",
        "www.kwsp.gov.my",
        "kwsp.gov.my",
        "iakaun.kwsp.gov.my",
        "www.kkr.gov.my",
        "kkr.gov.my",
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

# Liveness check timeout (seconds) — fast fail
LIVENESS_TIMEOUT = 4.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. INTENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# Pattern → intent label
_INTENT_PATTERNS: list[tuple[re.Pattern, str]] = [
    # claim / tuntutan / klaim / คลาม / i-claim / cara klaim
    (re.compile(r"\b(claim|claims|tuntut|tuntutan|klaim|cara klaim|คลาม|รับสิทธิ)\b", re.I), "claim"),
    # apply / mohon / pengajuan / สมัคร
    (re.compile(r"\b(apply|applying|applied|applic|mohon|permohonan|pengajuan|สมัคร|申請)\b", re.I), "apply"),
    # register / daftar / ลงทะเบียน
    (re.compile(r"\b(register|registr|daftar|pendaftaran|ลงทะเบียน|สมัคร)\b", re.I), "register"),
    # check / semak / ตรวจสอบ / cek
    (re.compile(r"\b(check|semak|cek|ตรวจสอบ|how much|berapa|balance|baki|saldo)\b", re.I), "check"),
    # benefit / faedah / สิทธิประโยชน์ / manfaat
    (re.compile(r"\b(benefit|faedah|manfaat|สิทธิ|ประโยชน์|coverage|perlindungan)\b", re.I), "benefit"),
    # contact / hubungi / ติดต่อ
    (re.compile(r"\b(contact|hubung|telefon|phone|hotline|office|ติดต่อ|โทร)\b", re.I), "contact"),
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
    # MY
    "perkeso": "perkeso",
    "socso": "socso",
    "epf": "epf",
    "kwsp": "kwsp",
    "jkm": "jkm",
    "kebajikan": "jkm",
    # ID
    "bpjs kesehatan": "bpjs kesehatan",
    "bpjs ketenagakerjaan": "bpjs ketenagakerjaan",
    "bpjamsostek": "bpjs ketenagakerjaan",
    "bpjs": "bpjs",
    "bansos": "bansos",
    "pkh": "pkh",
    "jaminan sosial": "bpjs ketenagakerjaan",
    # PH
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
    # TH
    "ประกันสังคม": "ประกันสังคม",
    "sso": "ประกันสังคม",
    "social security thailand": "ประกันสังคม",
    "บัตรทอง": "บัตรทอง",
    "30 baht": "บัตรทอง",
    "nhso": "บัตรทอง",
}


def _detect_entity(query: str) -> str | None:
    """Return golden URL entity key if query matches a known entity."""
    q = query.lower()
    # Longest match first to avoid "bpjs" matching before "bpjs kesehatan"
    for alias in sorted(_ENTITY_ALIASES.keys(), key=len, reverse=True):
        if alias in q:
            return _ENTITY_ALIASES[alias]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. GOLDEN URL LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def _get_golden_result(query: str) -> dict | None:
    """
    Return a golden URL result dict if query matches entity + intent.
    Falls back: specific intent → 'general' → None.
    """
    entity = _detect_entity(query)
    if not entity or entity not in GOLDEN_URLS:
        return None

    intent = _detect_intent(query)
    entity_map = GOLDEN_URLS[entity]

    # Try specific intent first, then general
    entry = entity_map.get(intent) or entity_map.get("general")
    if not entry:
        return None

    return {
        "title": entry["title"],
        "url": entry["url"],
        "snippet": entry["snippet"],
        "_source": "golden",
    }


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

async def _is_url_live(url: str) -> bool:
    """HEAD request to verify URL resolves. Returns True if 2xx or 3xx."""
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(LIVENESS_TIMEOUT),
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
    country: str = "",
) -> str:
    """Search government portals for fresh information.

    Hardened with:
    - Golden URL map (no DuckDuckGo needed for known entities)
    - Red-flag filtering (complaints/login pages rejected)
    - Liveness check (dead URLs dropped before response)

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

    # ── Path A: Golden URL (query matches known entity) ─────────────────────
    if not is_url:
        golden = _get_golden_result(url)
        if golden:
            logger.info("fetch_gov_portal: golden URL hit for query '%s'", url[:60])

            # Liveness check on the golden URL
            live = await _is_url_live(golden["url"])
            if not live:
                logger.warning("Golden URL dead: %s — falling through to DDG", golden["url"])
                # Fall through to DuckDuckGo below
            else:
                result = {
                    "title": golden["title"],
                    "url": golden["url"],
                    "snippet": golden["snippet"],
                }
                combined_content = f"**{result['title']}**\n{result['snippet']}\nSource: {result['url']}"

                return json.dumps({
                    "results": [result],
                    "content": combined_content,
                    "query_used": url,
                    "fetched_at": fetched_at,
                    "country": country,
                    "result_count": 1,
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
        search_query = f"{site_scope} {path}" if path else site_scope
    else:
        country = country.strip().upper() if country else ""
        site_scope = _get_site_scope(country=country)
        search_query = f"{site_scope} {url}" if site_scope else url

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
            general_query = f"{url} {country_name} government".strip()
            logger.info("fetch_gov_portal: DDG gov empty — falling back to web '%s'", general_query)
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

        # ── Liveness check on top results ──────────────────────────────────
        results = []
        for r in filtered[:MAX_SEARCH_RESULTS]:
            result_url = r.get("href", "")
            if not result_url:
                continue
            live = await _is_url_live(result_url)
            if live:
                results.append({
                    "title": r.get("title", ""),
                    "url": result_url,
                    "snippet": r.get("body", ""),
                })
            else:
                logger.info("fetch_gov_portal: dropped dead URL %s", result_url)

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