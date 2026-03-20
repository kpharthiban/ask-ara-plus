"""
AskAra+ MCP Tool Implementations
=================================
Each module contains one tool function that the FastMCP server exposes
to the LangChain ReAct agent. 

Ownership:
- search, language        → Pharthiban (Day 2)
- simplify, summarize,
  complexity              → Lineysha fills NLP logic (Day 3)
- translate, dialect      → Lineysha fills NLP logic (Day 4)
- scanner, profiler       → Pharthiban (Day 5)
- portal                  → Pharthiban (Tier 3, if time)
- speech                  → May move to frontend (Web Speech API)
"""

from tools.search import search_documents
from tools.language import detect_language
from tools.simplify import simplify_text
from tools.translate import translate_text
from tools.summarize import summarize_text
from tools.dialect import dialect_adapt
from tools.complexity import assess_complexity
from tools.speech import text_to_speech
from tools.scanner import scan_document
from tools.portal import fetch_gov_portal
from tools.profiler import profile_match

__all__ = [
    "search_documents",
    "detect_language",
    "simplify_text",
    "translate_text",
    "summarize_text",
    "dialect_adapt",
    "assess_complexity",
    "text_to_speech",
    "scan_document",
    "fetch_gov_portal",
    "profile_match",
]