/**
 * useWebSocket.ts — AskAra+ WebSocket Hook
 *
 * Key design: When the server sends a "structured" frame (step_cards or
 * recommendations), we BUFFER it — don't create a message yet. The LLM's
 * subsequent text tokens stream into a normal bubble as the intro text.
 * When "done" arrives, we MERGE the buffered structured data into that
 * same bubble, producing ONE unified message.
 *
 * Server protocol:
 *   { type: "token",      content: "..." }
 *   { type: "tool_start", content: "..." }
 *   { type: "tool_end",   content: "..." }
 *   { type: "done",       content: "..." }
 *   { type: "structured", content: { ... } }
 *   { type: "sources",    content: [...] }
 *   { type: "error",      content: "..." }
 *
 * Client sends:
 *   {
 *     message: "...",
 *     language?: "ms",
 *     country?: "MY",
 *     history?: [...],
 *     image_base64?: "...",   <- raw base64, no data-URI prefix
 *     image_media_type?: "image/jpeg",
 *   }
 *
 * Stability fixes (v2):
 *   - isMountedRef gates ALL reconnect scheduling so unmounting (React
 *     StrictMode, chatKey change) never causes a reconnect storm.
 *   - silentClose() nulls all WS handlers before calling .close() so the
 *     onclose callback cannot fire and schedule a reconnect on an instance
 *     we are intentionally discarding.
 *   - A 50 ms initial-connect delay absorbs React StrictMode dev-mode
 *     mount->unmount->mount; the phantom first-mount's timer is cancelled
 *     in cleanup before it ever calls connect().
 *
 * Persistence:
 *   - Accepts initialMessages (from localStorage, loaded by ChatWindow).
 *   - messages state is seeded from these on mount.
 *   - historyRef is reconstructed so the LLM retains prior-turn context.
 */

import { useState, useEffect, useRef, useCallback } from "react";
import type {
  Message,
  MessageContentType,
  Source,
  StructuredContent,
} from "@/lib/types";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
const WS_URL = `${BACKEND_URL.replace(/^http/, "ws")}/ws/chat`;

const MAX_RECONNECT_ATTEMPTS = 5;
const BASE_RECONNECT_DELAY_MS = 1000;

/**
 * Delay before the very first connect attempt (ms).
 * Long enough for React StrictMode's phantom unmount (~0 ms in practice) to
 * cancel the timer before it fires, but imperceptible to the user.
 */
const INITIAL_CONNECT_DELAY_MS = 50;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------
export type ConnectionStatus =
  | "connecting"
  | "connected"
  | "disconnected"
  | "error";

export interface SendOptions {
  language?: string;
  country?: string;
  /** Raw base64 image data (no data-URI prefix). */
  imageBase64?: string;
  /** MIME type of the image, e.g. "image/jpeg". Defaults to "image/jpeg". */
  imageMediaType?: string;
  /** Full data-URI used to render the thumbnail in the user bubble. */
  imagePreview?: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Null all event handlers then close a WebSocket.
 * Prevents the onclose callback from scheduling a reconnect after we have
 * already decided to stop (component unmount, new connection opening).
 */
function silentClose(ws: WebSocket | null): void {
  if (!ws) return;
  ws.onopen = null;
  ws.onmessage = null;
  ws.onerror = null;
  ws.onclose = null;
  if (
    ws.readyState === WebSocket.OPEN ||
    ws.readyState === WebSocket.CONNECTING
  ) {
    ws.close();
  }
}

/**
 * Reconstruct the LLM history array from persisted messages so Ara keeps
 * context across page reloads and chat switches.
 */
function buildHistoryFromMessages(
  messages: Message[]
): { role: string; content: string }[] {
  return messages
    .filter((m) => !m.isLoading && !m.isStreaming && m.content.trim() !== "")
    .map((m) => ({
      role: m.sender === "user" ? "user" : "assistant",
      content: m.content,
    }));
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * @param initialMessages  Persisted messages to restore on mount.
 *                         Passed by ChatWindow after reading localStorage.
 */
export function useWebSocket(initialMessages: Message[] = []) {
  // Seed state from persisted messages so the restored chat appears instantly.
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [isThinking, setIsThinking] = useState(false);
  const [activeTool, setActiveTool] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const initTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  /**
   * True while this hook instance is alive (mounted).
   * ALL reconnect scheduling is gated on this flag so an unmounting
   * component never queues a new connection attempt.
   */
  const isMountedRef = useRef(false);

  const streamingMsgId = useRef<string | null>(null);
  const toolsUsed = useRef<string[]>([]);

  // Seed history from persisted messages so the LLM retains context after
  // a page reload or chat switch (without this, Ara answers as if fresh).
  const historyRef = useRef<{ role: string; content: string }[]>(
    buildHistoryFromMessages(initialMessages)
  );

  // Cancel guard — prevents stale server frames from creating ghost bubbles
  const cancelledRef = useRef(false);

  // Buffered structured payload — merged into the message on "done"
  const pendingStructured = useRef<{
    contentType: MessageContentType;
    structured: StructuredContent;
  } | null>(null);

  // Buffered sources — merged into message on "done"
  const pendingSources = useRef<Source[]>([]);

  const genId = () =>
    `msg_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;

  // -------------------------------------------------------------------------
  // Message mutations
  // -------------------------------------------------------------------------

  const startAraBubble = useCallback(() => {
    const id = genId();
    streamingMsgId.current = id;

    const pending = pendingStructured.current;

    setMessages((prev) => [
      ...prev,
      {
        id,
        sender: "ara",
        content: "",
        contentType: pending ? pending.contentType : "text",
        timestamp: new Date(),
        isLoading: false,
        isStreaming: true,
        ...(pending ? { structured: pending.structured } : {}),
      },
    ]);
    return id;
  }, []);

  const appendToken = useCallback((token: string) => {
    const targetId = streamingMsgId.current;
    if (!targetId) return;
    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === targetId
          ? { ...msg, content: msg.content + token }
          : msg
      )
    );
  }, []);

  /**
   * Finalize the streaming message.
   * Merges buffered structured data (if any) into the same bubble so the
   * frontend renders ONE combined block (intro text + cards).
   */
  const finalizeMessage = useCallback((fullText: string) => {
    const targetId = streamingMsgId.current;
    if (!targetId) return;

    const pending = pendingStructured.current;
    const sources =
      pendingSources.current.length > 0
        ? [...pendingSources.current]
        : undefined;

    setMessages((prev) =>
      prev.map((msg) => {
        if (msg.id !== targetId) return msg;

        const base = {
          ...msg,
          content: fullText,
          isStreaming: false,
          isLoading: false,
          toolCalls:
            toolsUsed.current.length > 0
              ? [...toolsUsed.current]
              : undefined,
          ...(sources ? { sources } : {}),
        };

        if (pending) {
          return {
            ...base,
            contentType: pending.contentType,
            structured: pending.structured,
          };
        }

        return base;
      })
    );

    historyRef.current.push({ role: "assistant", content: fullText });
    streamingMsgId.current = null;
    pendingStructured.current = null;
    pendingSources.current = [];
    setIsThinking(false);
    setActiveTool(null);
  }, []);

  // -------------------------------------------------------------------------
  // Handle incoming server frames
  // -------------------------------------------------------------------------
  const handleServerMessage = useCallback(
    (event: MessageEvent) => {
      let data: { type: string; content?: any };
      try {
        data = JSON.parse(event.data);
      } catch {
        console.error("[ws] Bad JSON from server:", event.data);
        return;
      }

      switch (data.type) {
        case "token": {
          if (cancelledRef.current) break;
          if (!streamingMsgId.current) {
            startAraBubble();
            setIsThinking(false);
            setActiveTool(null);
          }
          if (typeof data.content === "string") {
            appendToken(data.content);
          }
          break;
        }

        case "tool_start": {
          if (cancelledRef.current) break;
          setIsThinking(true);
          const toolName =
            typeof data.content === "string" ? data.content : "tool";
          setActiveTool(toolName);
          toolsUsed.current.push(toolName);
          break;
        }

        case "tool_end": {
          if (cancelledRef.current) break;
          setActiveTool(null);
          break;
        }

        case "done": {
          if (cancelledRef.current) break;
          const fullText =
            typeof data.content === "string" ? data.content : "";
          if (!streamingMsgId.current) startAraBubble();
          finalizeMessage(fullText);
          break;
        }

        case "cancelled": {
          break;
        }

        case "structured": {
          const payload = data.content;
          if (payload && typeof payload === "object" && payload.type) {
            if (
              (payload.type === "step_cards" && Array.isArray(payload.cards)) ||
              (payload.type === "recommendations" &&
                Array.isArray(payload.items))
            ) {
              pendingStructured.current = {
                contentType: payload.type as MessageContentType,
                structured: payload,
              };

              const targetId = streamingMsgId.current;
              if (targetId) {
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === targetId
                      ? {
                          ...msg,
                          contentType: payload.type as MessageContentType,
                          structured: payload,
                        }
                      : msg
                  )
                );
              }
            }
          }
          break;
        }

        case "sources": {
          const sources = Array.isArray(data.content) ? data.content : [];
          pendingSources.current = sources;
          const targetId = streamingMsgId.current;
          if (targetId && sources.length > 0) {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === targetId ? { ...msg, sources } : msg
              )
            );
          }
          break;
        }

        case "error": {
          setIsThinking(false);
          setActiveTool(null);
          pendingStructured.current = null;
          pendingSources.current = [];
          if (streamingMsgId.current) {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === streamingMsgId.current
                  ? { ...msg, isStreaming: false, isLoading: false }
                  : msg
              )
            );
            streamingMsgId.current = null;
          }
          setMessages((prev) => [
            ...prev,
            {
              id: genId(),
              sender: "ara",
              content:
                typeof data.content === "string"
                  ? data.content
                  : "Maaf, something went wrong. Please try again.",
              contentType: "text",
              timestamp: new Date(),
            },
          ]);
          break;
        }

        default:
          console.warn("[ws] Unknown frame type:", data.type);
      }
    },
    [startAraBubble, appendToken, finalizeMessage]
  );

  // -------------------------------------------------------------------------
  // Connect
  // -------------------------------------------------------------------------
  const connect = useCallback(() => {
    // Hard guard: never open a socket for an unmounted component
    if (!isMountedRef.current) return;

    // Discard any existing socket WITHOUT triggering its onclose handler
    // (which would schedule another reconnect on the old instance).
    silentClose(wsRef.current);
    wsRef.current = null;

    setStatus("connecting");
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      if (!isMountedRef.current) { silentClose(ws); return; }
      setStatus("connected");
      reconnectAttempts.current = 0;
    };

    ws.onmessage = handleServerMessage;

    ws.onerror = () => {
      if (!isMountedRef.current) return;
      setStatus("error");
    };

    ws.onclose = () => {
      // If unmounted, bail immediately — do NOT schedule a reconnect
      if (!isMountedRef.current) return;

      setStatus("disconnected");
      setIsThinking(false);
      setActiveTool(null);

      if (reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
        const delay =
          BASE_RECONNECT_DELAY_MS * Math.pow(2, reconnectAttempts.current);
        reconnectTimer.current = setTimeout(() => {
          // Check again when the timer fires — component could have unmounted
          if (!isMountedRef.current) return;
          reconnectAttempts.current++;
          connect();
        }, delay);
      }
    };

    wsRef.current = ws;
  }, [handleServerMessage]);

  // -------------------------------------------------------------------------
  // Lifecycle — mount / unmount
  // -------------------------------------------------------------------------
  useEffect(() => {
    isMountedRef.current = true;

    // Short delay before the first connect attempt.
    //
    // React StrictMode (dev mode) runs: mount -> cleanup -> mount.
    // The phantom first-mount's timer (50 ms) is cancelled by the cleanup
    // before it fires, so connect() is never called for the phantom mount.
    // On the real second mount the timer runs normally (~50 ms latency,
    // imperceptible to the user).
    //
    // In production / non-StrictMode there is only one mount, so the timer
    // fires after 50 ms as normal.
    initTimer.current = setTimeout(() => {
      if (isMountedRef.current) connect();
    }, INITIAL_CONNECT_DELAY_MS);

    return () => {
      // Mark unmounted FIRST so no in-flight callback can schedule a reconnect
      isMountedRef.current = false;

      // Cancel any pending timers
      if (initTimer.current) clearTimeout(initTimer.current);
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);

      // Close the socket without firing onclose (silentClose nulls handlers)
      silentClose(wsRef.current);
      wsRef.current = null;
    };
  }, [connect]);

  // -------------------------------------------------------------------------
  // Send a message
  // -------------------------------------------------------------------------
  const sendMessage = useCallback(
    (content: string, options?: SendOptions) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        console.error("[ws] Not connected.");
        return;
      }

      const trimmed = content.trim();
      if (!trimmed && !options?.imageBase64) return;

      cancelledRef.current = false;

      setMessages((prev) => [
        ...prev,
        {
          id: genId(),
          sender: "user",
          content: trimmed,
          contentType: "text",
          timestamp: new Date(),
          ...(options?.imagePreview
            ? { imagePreview: options.imagePreview }
            : {}),
        },
      ]);

      setIsThinking(true);
      pendingStructured.current = null;
      pendingSources.current = [];
      toolsUsed.current = [];

      if (trimmed) {
        historyRef.current.push({ role: "user", content: trimmed });
      }

      const payload: Record<string, unknown> = {
        message: trimmed,
        language: options?.language || null,
        country: options?.country || null,
        history: historyRef.current.slice(0, trimmed ? -1 : undefined),
      };

      if (options?.imageBase64) {
        payload.image_base64 = options.imageBase64;
        payload.image_media_type = options.imageMediaType || "image/jpeg";
      }

      wsRef.current.send(JSON.stringify(payload));
    },
    []
  );

  // -------------------------------------------------------------------------
  // Stop generation
  // -------------------------------------------------------------------------
  const stopGeneration = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    cancelledRef.current = true;
    wsRef.current.send(JSON.stringify({ type: "cancel" }));

    const targetId = streamingMsgId.current;
    if (targetId) {
      setMessages((prev) => {
        const currentMsg = prev.find((m) => m.id === targetId);
        if (currentMsg?.content) {
          historyRef.current.push({
            role: "assistant",
            content: currentMsg.content,
          });
        }
        return prev.map((msg) =>
          msg.id === targetId
            ? { ...msg, isStreaming: false, isLoading: false }
            : msg
        );
      });
      streamingMsgId.current = null;
    }

    pendingStructured.current = null;
    pendingSources.current = [];
    toolsUsed.current = [];
    setIsThinking(false);
    setActiveTool(null);
  }, []);

  // -------------------------------------------------------------------------
  // Clear messages (new chat)
  // -------------------------------------------------------------------------
  const clearMessages = useCallback(() => {
    setMessages([]);
    historyRef.current = [];
    pendingStructured.current = null;
    pendingSources.current = [];
  }, []);

  return {
    messages,
    status,
    isThinking,
    activeTool,
    sendMessage,
    stopGeneration,
    clearMessages,
  };
}