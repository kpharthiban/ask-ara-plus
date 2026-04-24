/**
 * usePersistentChats.ts
 * ---------------------
 * Manages chat session list + per-chat message persistence via localStorage.
 *
 * Storage keys:
 *   "askara_chats"          → JSON array of ChatSession
 *   "askara_active_chat"    → active chat ID string
 *   "askara_msgs_{chatId}"  → JSON array of Message for that chat
 *
 * Place this file at:  src/hooks/usePersistentChats.ts
 * Import as:           @/hooks/usePersistentChats
 */

"use client";

import { useState, useCallback, useEffect } from "react";
import type { ChatSession } from "@/components/Sidebar";
import type { Message } from "@/lib/types";

// ── Storage keys ──────────────────────────────────────────────────────────
const CHATS_KEY = "askara_chats";
const ACTIVE_KEY = "askara_active_chat";
export const MSGS_PREFIX = "askara_msgs_";

// ── Safe localStorage wrappers ────────────────────────────────────────────

function safeGet(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function safeSet(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch (err) {
    console.warn("[AskAra] localStorage write failed:", err);
  }
}

function safeRemove(key: string): void {
  try {
    localStorage.removeItem(key);
  } catch {
    /* ignore */
  }
}

// ── Message serialization ─────────────────────────────────────────────────
// Strip transient UI fields before persisting.
// imagePreview (base64 data-URI) is omitted — it can be several MB per image
// and would blow through the ~5MB localStorage quota quickly.

function serializeMessages(messages: Message[]): string {
  return JSON.stringify(
    messages.map(({ isLoading, isStreaming, imagePreview, ...rest }) => ({
      ...rest,
      isLoading: false,
      isStreaming: false,
      timestamp:
        rest.timestamp instanceof Date
          ? rest.timestamp.toISOString()
          : rest.timestamp,
    }))
  );
}

function deserializeMessages(raw: string): Message[] {
  try {
    const arr: any[] = JSON.parse(raw);
    return arr.map((m) => ({
      ...m,
      timestamp: new Date(m.timestamp),
      isLoading: false as const,
      isStreaming: false as const,
    }));
  } catch {
    return [];
  }
}

// ── ChatSession serialization ─────────────────────────────────────────────

function serializeChats(chats: ChatSession[]): string {
  return JSON.stringify(
    chats.map((c) => ({
      ...c,
      createdAt:
        c.createdAt instanceof Date ? c.createdAt.toISOString() : c.createdAt,
      updatedAt:
        c.updatedAt instanceof Date ? c.updatedAt.toISOString() : c.updatedAt,
    }))
  );
}

function deserializeChats(raw: string): ChatSession[] {
  try {
    const arr: any[] = JSON.parse(raw);
    return arr.map((c) => ({
      ...c,
      createdAt: new Date(c.createdAt),
      updatedAt: c.updatedAt ? new Date(c.updatedAt) : undefined,
    }));
  } catch {
    return [];
  }
}

// ── Public message helpers (imported by ChatWindow) ───────────────────────

/** Load persisted messages for a given chat ID. Returns [] if none found. */
export function loadChatMessages(chatId: string): Message[] {
  if (typeof window === "undefined") return [];
  const raw = safeGet(`${MSGS_PREFIX}${chatId}`);
  if (!raw) return [];
  return deserializeMessages(raw);
}

/**
 * Persist the current messages for a chat.
 * Only writes if there's at least one non-transient, non-empty message.
 */
export function saveChatMessages(chatId: string, messages: Message[]): void {
  if (typeof window === "undefined" || !chatId) return;
  // Filter out loading placeholders and empty-content bubbles
  const toSave = messages.filter(
    (m) => !m.isLoading && !m.isStreaming && m.content !== ""
  );
  if (toSave.length === 0) return;
  safeSet(`${MSGS_PREFIX}${chatId}`, serializeMessages(toSave));
}

/** Remove persisted messages for a deleted chat. */
export function deleteChatMessages(chatId: string): void {
  if (typeof window === "undefined") return;
  safeRemove(`${MSGS_PREFIX}${chatId}`);
}

// ── Hook ──────────────────────────────────────────────────────────────────

export interface UsePersistentChatsReturn {
  /** True once localStorage has been read on the client. */
  isHydrated: boolean;
  chatHistory: ChatSession[];
  activeChatId: string | null;
  /** Create a new chat session, set it as active, and return its ID. */
  createChat: () => string;
  /** Switch to a different chat (triggers ChatWindow remount in page.tsx). */
  selectChat: (id: string) => void;
  /** Delete a chat session and its stored messages. */
  deleteChat: (id: string) => void;
  /** Update the title (and updatedAt) of a chat. Called after first message. */
  updateChatTitle: (id: string, title: string) => void;
}

export function usePersistentChats(): UsePersistentChatsReturn {
  const [isHydrated, setIsHydrated] = useState(false);
  const [chatHistory, setChatHistory] = useState<ChatSession[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);

  // ── Hydrate from localStorage once on the client ──────────────────────
  useEffect(() => {
    const rawChats = safeGet(CHATS_KEY);
    if (rawChats) {
      setChatHistory(deserializeChats(rawChats));
    }

    const rawActive = safeGet(ACTIVE_KEY);
    if (rawActive) {
      setActiveChatId(rawActive);
    }

    setIsHydrated(true);
  }, []);

  // ── Persist chatHistory whenever it changes (after hydration) ─────────
  useEffect(() => {
    if (!isHydrated) return;
    safeSet(CHATS_KEY, serializeChats(chatHistory));
  }, [isHydrated, chatHistory]);

  // ── Persist activeChatId whenever it changes (after hydration) ────────
  useEffect(() => {
    if (!isHydrated) return;
    if (activeChatId) {
      safeSet(ACTIVE_KEY, activeChatId);
    } else {
      safeRemove(ACTIVE_KEY);
    }
  }, [isHydrated, activeChatId]);

  // ── Actions ───────────────────────────────────────────────────────────

  const createChat = useCallback((): string => {
    const id = crypto.randomUUID();
    const now = new Date();
    const newSession: ChatSession = {
      id,
      title: "New Chat",
      createdAt: now,
      updatedAt: now,
    };
    setChatHistory((prev) => [newSession, ...prev]);
    setActiveChatId(id);
    return id;
  }, []);

  const selectChat = useCallback((id: string) => {
    setActiveChatId(id);
  }, []);

  const deleteChat = useCallback((id: string) => {
    deleteChatMessages(id);
    setChatHistory((prev) => prev.filter((c) => c.id !== id));
    setActiveChatId((prev) => (prev === id ? null : prev));
  }, []);

  const updateChatTitle = useCallback((id: string, title: string) => {
    const safe = title.trim();
    const truncated =
      safe.length > 60 ? safe.slice(0, 60) + "…" : safe;
    setChatHistory((prev) =>
      prev.map((c) =>
        c.id === id
          ? { ...c, title: truncated || "New Chat", updatedAt: new Date() }
          : c
      )
    );
  }, []);

  return {
    isHydrated,
    chatHistory,
    activeChatId,
    createChat,
    selectChat,
    deleteChat,
    updateChatTitle,
  };
}