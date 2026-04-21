"""
AskAra+ — FastAPI Backend Server
---------------------------------
Entrypoint for the backend. Provides:
  - GET  /health          → Health check (for Railway + monitoring)
  - POST /chat            → REST chat endpoint (deterministic agent pipeline)
  - WS   /ws/chat         → WebSocket for streaming chat (deterministic pipeline)
  - POST /api/transcribe  → Speech-to-Text via Groq Whisper
  - POST /api/tts         → Text-to-Speech via edge-tts

Run locally:
    cd backend
    uv run uvicorn server:app --reload --host 0.0.0.0 --port 8000

Architecture:
    Client → FastAPI (server.py)
               ├── /chat         → agent.py (deterministic pipeline) → tools/*.py → LLM → response
               ├── /ws/chat      → agent.py (streaming) → tools/*.py → LLM → streaming response
               ├── /api/transcribe → Groq Whisper
               └── /api/tts      → edge-tts
"""

from __future__ import annotations

import asyncio
import os
import json
import logging
import base64
import io
import httpx
from fastapi import UploadFile, File, Form
from fastapi.responses import Response

from datetime import datetime, timezone
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llm_client import call_llm, call_llm_streaming, close_client, LLMError
from db import get_collection

from agent import run_agent, run_agent_streaming, cleanup_agent, init_mcp_tools


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("askara.server")

# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AskAra+ backend starting up...")

    try:
        collection = get_collection()
        count = collection.count()
        logger.info("ChromaDB ready — %d chunks in collection.", count)
    except Exception as exc:
        logger.warning("ChromaDB check failed: %s (will retry on first query)", exc)

    # Connect to MCP server (mcp_server.py must be running on port 8001)
    # Falls back gracefully to direct Python imports if MCP is offline.
    await init_mcp_tools()
    logger.info("LangGraph ReAct agent ready (MCP or direct-import mode).")

    yield

    logger.info("AskAra+ backend shutting down...")
    await cleanup_agent()
    await close_client()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AskAra+",
    description="Multilingual ASEAN government services assistant",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

ALLOWED_ORIGINS: list[str] = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "https://ara.fineeagle.cc",
    "https://mcp-ara.fineeagle.cc",
]

_vercel_url = os.getenv("FRONTEND_URL", "")
if _vercel_url:
    ALLOWED_ORIGINS.append(_vercel_url)
    ALLOWED_ORIGINS.append(_vercel_url.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.fineeagle\.cc",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Request body for POST /chat."""
    message: str
    language: str | None = None
    country: str | None = None
    history: list[dict] | None = None
    image_base64: str | None = None
    image_media_type: str = "image/jpeg"

class ChatResponse(BaseModel):
    """Response body for POST /chat."""
    reply: str
    sources: list[dict] | None = None
    language_detected: dict | None = None
    tool_calls: list[str] | None = None
    structured: dict | None = None
    timestamp: str

# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway and uptime monitoring."""
    chroma_ok = False
    chroma_count = 0
    try:
        collection = get_collection()
        chroma_count = collection.count()
        chroma_ok = True
    except Exception:
        pass

    return {
        "status": "ok",
        "service": "askara-backend",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chromadb": {
            "connected": chroma_ok,
            "document_chunks": chroma_count,
        },
        "agent": {
            "type": "deterministic_pipeline",
        },
    }

# ---------------------------------------------------------------------------
# POST /chat — REST endpoint (through deterministic agent)
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat_rest(req: ChatRequest):
    """Non-streaming chat endpoint. Routes through the deterministic agent pipeline."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        result = await run_agent(
            req.message,
            country=req.country,
            language=req.language,
            history=req.history,
            image_base64=req.image_base64 or None,
            image_media_type=req.image_media_type,
        )

        return ChatResponse(
            reply=result["reply"],
            sources=result.get("sources"),
            language_detected=result.get("language_detected"),
            tool_calls=result.get("tool_calls"),
            structured=result.get("structured"),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as exc:
        logger.error("Chat failed: %s", exc, exc_info=True)

        # Fallback: direct LLM call if agent fails
        logger.warning("Falling back to direct LLM call.")
        try:
            fallback_reply = await call_llm(
                req.message,
                system_prompt="You are Ara, a helpful multilingual assistant for ASEAN government services.",
                history=req.history,
            )
            return ChatResponse(
                reply=fallback_reply,
                sources=None,
                language_detected=None,
                tool_calls=["fallback_direct_llm"],
                structured=None,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        except LLMError as llm_exc:
            raise HTTPException(status_code=502, detail=f"Service error: {llm_exc}")


# ---------------------------------------------------------------------------
# WS /ws/chat — Streaming WebSocket endpoint (through deterministic agent)
# ---------------------------------------------------------------------------

@app.websocket("/ws/chat")
async def chat_websocket(ws: WebSocket):
    """
    WebSocket endpoint for streaming chat via the deterministic agent pipeline.

    Client sends JSON:
        { "message": "...", "language": "ms", "country": "MY", "history": [...] }

    Server streams back JSON frames:
        { "type": "token",      "content": "..." }
        { "type": "tool_start", "content": "..." }
        { "type": "tool_end",   "content": "..." }
        { "type": "structured", "content": {...} }
        { "type": "done",       "content": "..." }
        { "type": "sources",    "content": [...] }
        { "type": "error",      "content": "..." }

    Disconnect handling (v2):
        - WebSocketDisconnect raised by ws.send_json() during streaming is now
          caught and re-raised immediately rather than being swallowed by the
          generic except Exception clause and triggering a pointless LLM
          fallback on an already-dead connection.
        - safe_send() wraps every ws.send_json() call: on client disconnect it
          logs a single INFO line and returns False, letting the caller break
          out of the streaming loop cleanly without a noisy traceback.
    """
    await ws.accept()
    logger.info("WebSocket client connected.")

    cancel_event: asyncio.Event | None = None

    async def safe_send(payload: dict) -> bool:
        """
        Send a JSON frame to the client.
        Returns True on success, False if the client has disconnected.
        Raises nothing — all disconnect exceptions are absorbed here so the
        caller can break/return without a cascading traceback.
        """
        try:
            await ws.send_json(payload)
            return True
        except WebSocketDisconnect:
            return False
        except Exception:
            # Covers uvicorn's ClientDisconnected and any other send failure
            return False

    try:
        while True:
            raw = await ws.receive_text()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await safe_send({"type": "error", "content": "Invalid JSON."})
                continue

            # Handle cancel request from client
            if data.get("type") == "cancel":
                if cancel_event:
                    cancel_event.set()
                    logger.info("Client requested cancel.")
                continue

            message = data.get("message", "").strip()
            image_base64 = data.get("image_base64") or None
            image_media_type = data.get("image_media_type") or "image/jpeg"

            # Require at least a text message or an image
            if not message and not image_base64:
                await safe_send({"type": "error", "content": "Empty message."})
                continue

            country = data.get("country")
            language = data.get("language")
            history = data.get("history")

            cancel_event = asyncio.Event()

            try:
                full_response = ""
                cancelled = False
                client_gone = False

                async for event in run_agent_streaming(
                    message,
                    country=country,
                    language=language,
                    history=history,
                    image_base64=image_base64,
                    image_media_type=image_media_type,
                ):
                    if cancel_event.is_set():
                        cancelled = True
                        break

                    if event.get("type") == "done":
                        full_response = event.get("content", "")

                    # If client already gone, stop streaming silently
                    if not await safe_send(event):
                        client_gone = True
                        break

                if cancelled and not client_gone:
                    await safe_send({"type": "cancelled", "content": ""})
                    logger.info("Generation cancelled by client.")

                if client_gone:
                    # Client disconnected mid-stream — exit the handler
                    logger.info("Client disconnected mid-stream, aborting.")
                    return

            except WebSocketDisconnect:
                # Client disconnected before/during agent processing — exit cleanly
                logger.info("WebSocket client disconnected during agent run.")
                return

            except Exception as exc:
                # Genuine agent error (not a client disconnect) — try LLM fallback
                logger.error("WebSocket agent error: %s", exc, exc_info=True)
                logger.warning("Agent failed, falling back to direct LLM streaming.")

                try:
                    full_response = ""
                    async for token in call_llm_streaming(
                        message,
                        system_prompt=(
                            "You are Ara, a helpful multilingual assistant "
                            "for ASEAN government services."
                        ),
                        history=history,
                    ):
                        if cancel_event and cancel_event.is_set():
                            break
                        full_response += token
                        if not await safe_send({"type": "token", "content": token}):
                            logger.info("Client disconnected during fallback stream.")
                            return

                    await safe_send({"type": "done", "content": full_response})

                except LLMError:
                    await safe_send({
                        "type": "error",
                        "content": "Sorry, I'm having trouble right now. Please try again.",
                    })

            finally:
                cancel_event = None

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")

    except Exception as exc:
        logger.error("WebSocket unexpected error: %s", exc)
        try:
            await ws.send_json({"type": "error", "content": "Internal server error."})
            await ws.close()
        except Exception:
            pass


WHISPER_PROMPTS = {
    "ms": "Bahasa Melayu. Contoh: MySejahtera, KWSP, EPF, PERKESO, SOCSO, JPN, bantuan, pendaftaran, permohonan, kad pengenalan, MyKad.",
    "id": "Bahasa Indonesia. Contoh: BPJS, KTP, NIK, bansos, pendaftaran, permohonan, kartu keluarga, puskesmas, kecamatan.",
    "fil": "Filipino Tagalog. Halimbawa: PhilHealth, SSS, Pag-IBIG, barangay, munisipyo, serbisyo, aplikasyon, benepisyo.",
    "th": "ภาษาไทย ตัวอย่าง: บัตรประชาชน สำนักงานเขต ประกันสังคม สวัสดิการ ลงทะเบียน",
    "en": "ASEAN government services: MySejahtera, EPF, SOCSO, BPJS, PhilHealth, SSS, Pag-IBIG, registration, application, benefit.",
}

DEFAULT_PROMPT = "ASEAN government services, multilingual: Malay, Indonesian, Filipino, Thai, English."


# ---------------------------------------------------------------------------
# POST /api/transcribe — Speech-to-Text via Groq Whisper
# ---------------------------------------------------------------------------

@app.post("/api/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = Form(default=""),
):
    """Transcribe audio using Groq Whisper API (whisper-large-v3)."""

    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not configured. Add it to your .env file.",
        )

    audio_bytes = await audio.read()
    if len(audio_bytes) < 1000:
        raise HTTPException(status_code=400, detail="Audio too short.")

    filename = audio.filename or "recording.webm"

    prompt = WHISPER_PROMPTS.get(language, DEFAULT_PROMPT)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            files = {
                "file": (filename, audio_bytes, audio.content_type or "audio/webm"),
            }
            data = {
                "model": "whisper-large-v3",
                "prompt": prompt,
            }
            if language:
                data["language"] = language

            response = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files=files,
                data=data,
            )

        if response.status_code != 200:
            logger.error("Groq Whisper error %d: %s", response.status_code, response.text)
            raise HTTPException(
                status_code=502,
                detail="Transcription service error. Please try again.",
            )

        result = response.json()
        transcript = result.get("text", "").strip()

        logger.info("Transcribed (%s): %s", language or "auto", transcript[:80])
        return {"text": transcript}

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Transcription timed out.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Transcription failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Transcription failed.")


# ---------------------------------------------------------------------------
# POST /api/tts — Text-to-Speech via edge-tts (free, no API key)
# ---------------------------------------------------------------------------

EDGE_TTS_VOICES: dict[str, str] = {
    "ms":  "ms-MY-YasminNeural",
    "id":  "id-ID-GadisNeural",
    "th":  "th-TH-PremwadeeNeural",
    "fil": "fil-PH-BlessicaNeural",
    "tl":  "fil-PH-BlessicaNeural",
    "en":  "en-US-JennyNeural",
}

DEFAULT_TTS_VOICE = "en-US-JennyNeural"


class TTSRequest(BaseModel):
    text: str
    language: str = "en"


@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    if len(text) > 2000:
        text = text[:2000]

    lang_base = req.language.split("-")[0].lower() if req.language else "en"
    voice = EDGE_TTS_VOICES.get(lang_base, DEFAULT_TTS_VOICE)

    try:
        import edge_tts

        communicate = edge_tts.Communicate(text, voice)
        buffer = io.BytesIO()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.write(chunk["data"])

        audio_bytes = buffer.getvalue()

        if not audio_bytes:
            raise HTTPException(status_code=500, detail="TTS produced no audio.")

        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        logger.info("TTS complete: lang=%s voice=%s size=%d bytes", lang_base, voice, len(audio_bytes))

        return {
            "audio_base64": audio_b64,
            "content_type": "audio/mpeg",
            "voice": voice,
            "language": lang_base,
        }

    except ImportError:
        logger.warning("edge-tts not installed — TTS unavailable.")
        raise HTTPException(
            status_code=501,
            detail="TTS not available — edge-tts package not installed.",
        )
    except Exception as exc:
        logger.error("TTS failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Text-to-speech failed.")


# /api/scan endpoint removed — image analysis is now handled directly by
# the vision model (Gemma-SEA-LION-v4-27B-IT) via the WebSocket chat pipeline.