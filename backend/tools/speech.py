"""
text_to_speech — Convert text to speech audio.

Owner: Pharthiban (low priority — likely handled on frontend instead)
Note: Browser Web Speech API (speechSynthesis) is free and works offline.
      This server-side tool is the fallback if browser TTS quality is poor.

Options:
  - gTTS (Google Text-to-Speech) — free, simple, needs network
  - Browser speechSynthesis — handled entirely on frontend (preferred)
  - Edge TTS — free, better quality, async
"""

import json


def text_to_speech(
    text: str,
    language: str = "en",
) -> str:
    """Convert text to speech audio for users who prefer listening.

    NOTE: The frontend already supports browser-based TTS via Web Speech API.
    Only call this server-side tool if the frontend TTS fails or if a
    higher-quality voice is needed for the demo.

    Args:
        text: The text to convert to speech (keep under 500 chars for best results).
        language: Language code for voice selection ("ms", "en", "id", "tl", "th").

    Returns:
        JSON string:
        {
            "audio_url": "data:audio/mp3;base64,..." or "https://...",
            "duration_seconds": 8.5,
            "language": "ms",
            "engine": "gtts" | "browser" | "edge_tts"
        }
    """
    # ── STUB: likely stays as stub if frontend TTS is sufficient ─
    # from gtts import gTTS
    # import base64, io
    #
    # tts = gTTS(text=text, lang=language)
    # buffer = io.BytesIO()
    # tts.write_to_fp(buffer)
    # audio_b64 = base64.b64encode(buffer.getvalue()).decode()
    # ─────────────────────────────────────────────────────────────

    stub_result = {
        "audio_url": "",
        "duration_seconds": 0,
        "language": language,
        "engine": "browser",
        "note": "TTS handled on frontend via Web Speech API. This tool is a fallback.",
    }
    return json.dumps(stub_result, ensure_ascii=False)