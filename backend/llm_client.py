"""
LLM Client Wrapper for AskAra+
-------------------------------
Unified async interface to SEA-LION (primary) via OpenAI-compatible API.
Every module that needs LLM completions imports from here.

Usage:
    from llm_client import call_llm, call_llm_streaming

    # Simple completion
    response = await call_llm("What is SOCSO?", system_prompt="You are Ara...")

    # Streaming (yields token strings)
    async for token in call_llm_streaming("Explain EPF", system_prompt="..."):
        print(token, end="")
"""

from __future__ import annotations

import os
import json
import logging
from typing import AsyncGenerator

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("askara.llm")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# SEA-LION (Primary) — OpenAI-compatible endpoint
SEALION_API_KEY: str = os.getenv("SEALION_API_KEY", "")
SEALION_API_BASE: str = os.getenv("SEALION_API_BASE", "https://api.sea-lion.ai/v1")
SEALION_MODEL: str = os.getenv("SEALION_MODEL", "aisingapore/Qwen-SEA-LION-v4-32B-IT")
# Vision: Gemma-SEA-LION-v4-27B-IT is the multimodal model for Snap & Understand.
# Qwen-SEA-LION is text-only; Gemma-SEA-LION handles both text + image.
SEALION_VL_MODEL: str = os.getenv("SEALION_VL_MODEL", "aisingapore/Gemma-SEA-LION-v4-27B-IT")

# Request defaults
DEFAULT_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
DEFAULT_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "60.0"))

# Reusable async HTTP client (connection pooling)
_http_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Lazy singleton for the async HTTP client."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(LLM_TIMEOUT, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _http_client


def _build_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _build_messages(
    prompt: str,
    system_prompt: str | None = None,
    history: list[dict] | None = None,
) -> list[dict]:
    """
    Build the messages array for the chat completions endpoint.

    Args:
        prompt: The user's current message.
        system_prompt: Optional system instructions for Ara.
        history: Optional prior conversation turns
                 [{"role": "user"|"assistant", "content": "..."}]
    """
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    return messages


# ---------------------------------------------------------------------------
# Core: Non-streaming completion
# ---------------------------------------------------------------------------

async def call_llm(
    prompt: str,
    *,
    system_prompt: str | None = None,
    history: list[dict] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> str:
    """
    Send a chat completion request and return the full response text.

    Args:
        prompt: User message.
        system_prompt: System-level instructions.
        history: Prior conversation turns.
        model: Override model name (defaults to SEALION_MODEL).
        temperature: Sampling temperature (defaults to 0.3).
        max_tokens: Max tokens in response.
        json_mode: If True, request JSON output format.

    Returns:
        The assistant's response text.

    Raises:
        LLMError: On API failure after all retries.
    """
    client = _get_client()
    model = model or SEALION_MODEL
    temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
    max_tokens = max_tokens or DEFAULT_MAX_TOKENS

    payload: dict = {
        "model": model,
        "messages": _build_messages(prompt, system_prompt, history),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    url = f"{SEALION_API_BASE}/chat/completions"
    headers = _build_headers(SEALION_API_KEY)

    logger.debug("LLM request → %s  model=%s  tokens=%d", url, model, max_tokens)

    try:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        logger.debug("LLM response: %d chars", len(content))
        return content.strip()

    except httpx.HTTPStatusError as exc:
        logger.error(
            "LLM API error %d: %s",
            exc.response.status_code,
            exc.response.text[:500],
        )
        raise LLMError(
            f"SEA-LION API returned {exc.response.status_code}"
        ) from exc

    except httpx.TimeoutException as exc:
        logger.error("LLM request timed out after %.1fs", LLM_TIMEOUT)
        raise LLMError("LLM request timed out") from exc

    except Exception as exc:
        logger.error("Unexpected LLM error: %s", exc)
        raise LLMError(f"LLM call failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Core: Streaming completion
# ---------------------------------------------------------------------------

async def call_llm_streaming(
    prompt: str,
    *,
    system_prompt: str | None = None,
    history: list[dict] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream chat completion tokens as they arrive.

    Yields:
        Individual token strings from the LLM response.

    Usage:
        async for token in call_llm_streaming("Hello"):
            print(token, end="", flush=True)
    """
    client = _get_client()
    model = model or SEALION_MODEL
    temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
    max_tokens = max_tokens or DEFAULT_MAX_TOKENS

    payload: dict = {
        "model": model,
        "messages": _build_messages(prompt, system_prompt, history),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    url = f"{SEALION_API_BASE}/chat/completions"
    headers = _build_headers(SEALION_API_KEY)

    try:
        async with client.stream(
            "POST", url, headers=headers, json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    except httpx.HTTPStatusError as exc:
        logger.error("LLM stream error %d: %s", exc.response.status_code, exc.response.text[:500])
        raise LLMError(f"SEA-LION stream returned {exc.response.status_code}") from exc

    except httpx.TimeoutException as exc:
        logger.error("LLM stream timed out")
        raise LLMError("LLM stream timed out") from exc


# ---------------------------------------------------------------------------
# Core: Vision model completion (for Snap & Understand)
# ---------------------------------------------------------------------------

async def call_llm_vision(
    image_base64: str,
    prompt: str,
    *,
    system_prompt: str | None = None,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    image_media_type: str = "image/jpeg",
) -> str:
    """
    Send an image + text prompt to the vision model (OpenAI-compatible format).

    Uses Gemma-SEA-LION-v4-27B-IT (multimodal) — the Qwen variant is text-only.
    Same chat completions endpoint but with image_url content blocks.

    Args:
        image_base64: Base64-encoded image data (no data URI prefix).
        prompt: Text prompt to accompany the image.
        system_prompt: Optional system instructions.
        model: Override model (defaults to SEALION_VL_MODEL — Gemma-SEA-LION multimodal).
        temperature: Sampling temperature.
        max_tokens: Max response tokens.
        image_media_type: MIME type of the image ("image/jpeg" or "image/png").

    Returns:
        The model's response text.

    Raises:
        LLMError: On API failure.
    """
    client = _get_client()
    model = model or SEALION_VL_MODEL
    
    # Strip data URI prefix if accidentally included
    if image_base64.startswith("data:"):
        # "data:image/jpeg;base64,/9j/4AAQ..." → "/9j/4AAQ..."
        image_base64 = image_base64.split(",", 1)[1]
        
    # Build messages with image content (OpenAI Vision API format)
    user_content: list[dict] = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{image_media_type};base64,{image_base64}",
            },
        },
        {
            "type": "text",
            "text": prompt,
        },
    ]

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    url = f"{SEALION_API_BASE}/chat/completions"
    headers = _build_headers(SEALION_API_KEY)

    logger.debug("VL request → %s  model=%s  tokens=%d", url, model, max_tokens)

    try:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        logger.debug("VL response: %d chars", len(content))
        return content.strip()

    except httpx.HTTPStatusError as exc:
        logger.error(
            "VL API error %d: %s",
            exc.response.status_code,
            exc.response.text[:500],
        )
        raise LLMError(
            f"Vision API returned {exc.response.status_code}"
        ) from exc

    except httpx.TimeoutException as exc:
        logger.error("VL request timed out after %.1fs", LLM_TIMEOUT)
        raise LLMError("Vision request timed out") from exc

    except Exception as exc:
        logger.error("Unexpected VL error: %s", exc)
        raise LLMError(f"Vision call failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Convenience: structured JSON output
# ---------------------------------------------------------------------------

async def call_llm_json(
    prompt: str,
    *,
    system_prompt: str | None = None,
    history: list[dict] | None = None,
    model: str | None = None,
) -> dict:
    """
    Call the LLM expecting a JSON response. Parses and returns a dict.
    Falls back to extracting JSON from markdown code fences if needed.
    """
    raw = await call_llm(
        prompt,
        system_prompt=system_prompt,
        history=history,
        model=model,
        json_mode=True,
        temperature=0.1,  # Lower temp for structured output
    )

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fallback: extract from ```json ... ``` fences
    import re
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    raise LLMError(f"Failed to parse JSON from LLM response: {raw[:200]}")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

async def close_client() -> None:
    """Close the HTTP client gracefully. Call on app shutdown."""
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None
        logger.info("LLM HTTP client closed.")


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Raised when an LLM API call fails."""
    pass