"""
fetch_gov_portal — Search government portals for fresh information.

Owner: Pharthiban
Depends on: duckduckgo-search (pip install duckduckgo-search)

Approach:
  - Uses DuckDuckGo search with site: operator to scope results to
    allowlisted government domains only.
  - No direct scraping — avoids 404s, JS-rendering issues, SSL cert problems.
  - Returns search snippets + URLs so the agent can cite sources.
  - Falls back gracefully if DuckDuckGo rate-limits or fails.

Security: Only allowlisted government domains are permitted.
The site: operator ensures DuckDuckGo only returns results from gov sites.
"""

import json
import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

logger = logging.getLogger("askara.portal")


# ── Allowlisted government domains ────────────────────────────
ALLOWED_DOMAINS: dict[str, list[str]] = {
    "MY": [
        "www.perkeso.gov.my",
        "perkeso.gov.my",
        "www.mohr.gov.my",
        "www.jkm.gov.my",
        "www.moh.gov.my",
        "www.kkr.gov.my",
    ],
    "ID": [
        "www.bpjs-kesehatan.go.id",
        "bpjs-kesehatan.go.id",
        "kemensos.go.id",
        "www.kemnaker.go.id",
        "bp2mi.go.id",
    ],
    "PH": [
        "www.dmw.gov.ph",
        "www.owwa.gov.ph",
        "www.dti.gov.ph",
        "www.dswd.gov.ph",
    ],
    "TH": [
        "www.sso.go.th",
        "www.mol.go.th",
        "www.doe.go.th",
    ],
    "ASEAN": [
        "asean.org",
        "www.asean.org",
    ],
}

# Map country codes to their gov domain TLDs for site: scoping
COUNTRY_SITE_SCOPES: dict[str, str] = {
    "MY": "site:gov.my",
    "ID": "site:go.id",
    "PH": "site:gov.ph",
    "TH": "site:go.th",
    "ASEAN": "site:asean.org",
}

MAX_SEARCH_RESULTS = 5


def _is_allowed(url: str) -> bool:
    """Check if URL domain is in the allowlist."""
    domain = urlparse(url).netloc.lower()
    domain_no_www = domain.removeprefix("www.")
    for domains in ALLOWED_DOMAINS.values():
        if domain in domains or f"www.{domain_no_www}" in domains:
            return True
    return False


def _get_site_scope(url: str = "", country: str = "") -> str:
    """Get the DuckDuckGo site: scope for the query.

    If a specific URL is given, scope to that exact domain.
    Otherwise, scope to the country's gov TLD.
    """
    if url:
        parsed = urlparse(url if url.startswith("http") else f"https://{url}")
        domain = parsed.netloc.lower()
        if domain:
            return f"site:{domain}"

    if country and country.upper() in COUNTRY_SITE_SCOPES:
        return COUNTRY_SITE_SCOPES[country.upper()]

    # Default: no scope (will search broadly)
    return ""


def _detect_country(url: str) -> str:
    """Detect country from the domain TLD."""
    domain = urlparse(url).netloc.lower() if url.startswith("http") else url.lower()
    if ".my" in domain:
        return "MY"
    elif ".id" in domain:
        return "ID"
    elif ".ph" in domain:
        return "PH"
    elif ".th" in domain:
        return "TH"
    return ""


async def fetch_gov_portal(
    url: str,
    country: str = "",
) -> str:
    """Search government portals for fresh information.

    Use this tool when the knowledge base doesn't have the answer and
    the user asks about a specific government service, or when information
    may have been updated since the documents were collected.

    SECURITY: Only searches allowlisted government domains.
    Results are scoped using DuckDuckGo's site: operator.

    Args:
        url: A government portal URL to scope the search, OR a search query.
             If it looks like a URL, the search is scoped to that domain.
             If it looks like a query, it searches across gov sites for that country.
        country: Country code ("MY", "ID", "PH", "TH") to scope search.

    Returns:
        JSON string:
        {
            "results": [
                {
                    "title": "Page title",
                    "url": "https://...",
                    "snippet": "Page summary text..."
                }
            ],
            "content": "Combined text from all results...",
            "query_used": "site:gov.my SOCSO registration",
            "fetched_at": "2026-03-13T12:00:00Z",
            "country": "MY",
            "result_count": 3,
            "status": "success" | "no_results" | "error"
        }
    """
    # ── Input validation ──────────────────────────────────────
    if not url or not url.strip():
        return json.dumps({
            "results": [],
            "query_used": "",
            "fetched_at": "",
            "country": "",
            "result_count": 0,
            "status": "error",
            "error": "No URL or search query provided.",
        })

    url = url.strip()
    fetched_at = datetime.now(timezone.utc).isoformat()

    # ── Determine if input is a URL or a search query ─────────
    is_url = url.startswith("http") or url.startswith("www.") or "gov." in url.lower()

    if is_url:
        # Normalize URL
        clean_url = url if url.startswith("http") else f"https://{url}"

        # Security: check allowlist
        if not _is_allowed(clean_url):
            domain = urlparse(clean_url).netloc
            return json.dumps({
                "results": [],
                "query_used": "",
                "fetched_at": fetched_at,
                "country": "",
                "result_count": 0,
                "status": "blocked",
                "error": (
                    f"Domain '{domain}' is not in the allowlist. "
                    f"Only government portals from MY, ID, PH, TH, and ASEAN are permitted."
                ),
            })

        if not country:
            country = _detect_country(clean_url)

        # Build search query: scope to the specific domain
        site_scope = _get_site_scope(url=clean_url)
        # Extract a useful search term from the URL path
        path = urlparse(clean_url).path.strip("/").replace("-", " ").replace("/", " ")
        search_query = f"{site_scope} {path}" if path else site_scope

    else:
        # Input is a search query, not a URL
        country = country.strip().upper() if country else ""
        site_scope = _get_site_scope(country=country)
        search_query = f"{site_scope} {url}" if site_scope else url

    # ── Run DuckDuckGo search ─────────────────────────────────
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return json.dumps({
                "results": [],
                "query_used": search_query,
                "fetched_at": fetched_at,
                "country": country,
                "result_count": 0,
                "status": "error",
                "error": "DuckDuckGo search not installed. Run: uv add ddgs",
            })

    try:
        ddgs = DDGS()

        # ── Pass 1: Search gov domains only (trusted) ─────────
        logger.info("fetch_gov_portal: searching gov domains — '%s'", search_query)

        raw_results = ddgs.text(
            search_query,
            max_results=MAX_SEARCH_RESULTS,
        )

        source_tier = "government"  # Track where results came from

        # ── Pass 2: If gov search returned nothing, try general web ──
        if not raw_results and not is_url:
            # Strip the site: scope and search the open web
            general_query = url.strip()  # Original user query without site: prefix
            if country:
                country_name = {
                    "MY": "Malaysia", "ID": "Indonesia",
                    "PH": "Philippines", "TH": "Thailand",
                }.get(country.upper(), "")
                if country_name:
                    general_query = f"{general_query} {country_name} government"

            logger.info(
                "fetch_gov_portal: gov search empty — falling back to general web — '%s'",
                general_query,
            )

            raw_results = ddgs.text(
                general_query,
                max_results=MAX_SEARCH_RESULTS,
            )
            source_tier = "web"

        if not raw_results:
            return json.dumps({
                "results": [],
                "query_used": search_query,
                "fetched_at": fetched_at,
                "country": country,
                "result_count": 0,
                "status": "no_results",
                "note": "No results found from government portals or general web search.",
            })

        # Format results
        results = []
        for r in raw_results:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })

        # Combine all snippets into a content field for the agent
        combined_content = "\n\n".join(
            f"**{r['title']}**\n{r['snippet']}\nSource: {r['url']}"
            for r in results
        )

        logger.info(
            "fetch_gov_portal: found %d results (%s) for '%s'",
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
            "results": [],
            "query_used": search_query,
            "fetched_at": fetched_at,
            "country": country,
            "result_count": 0,
            "status": "error",
            "error": f"Search failed ({error_name}): {str(e)}",
        })