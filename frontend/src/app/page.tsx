"use client";

import { useState, useCallback, useEffect } from "react";
import ChatWindow from "@/components/ChatWindow";
import Sidebar, { type ChatSession } from "@/components/Sidebar";
import { MessageSquarePlus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { usePersistentChats } from "@/hooks/usePersistentChats";

export default function Home() {
  // ── Persistent chat state ───────────────────────────────────────────────
  // usePersistentChats manages chatHistory + activeChatId in localStorage.
  // On first load it hydrates from storage, restoring the last active session.
  const {
    isHydrated,
    chatHistory,
    activeChatId,
    createChat,
    selectChat,
    deleteChat,
    updateChatTitle,
  } = usePersistentChats();

  // ── Chat remount key ────────────────────────────────────────────────────
  // chatKey is the React key on <ChatWindow>. Changing it forces a full
  // remount so useWebSocket reinitialises with the correct initialMessages
  // for the selected chat.
  //
  // Rule:
  //   • New Chat button   → bump chatKey (fresh empty window)
  //   • Select Chat       → bump chatKey (load different chat's messages)
  //   • First-message auto-create → do NOT bump chatKey (messages already
  //     live in the current window; we only assign an ID so saves work)
  //   • Delete active chat → bump chatKey (clear the window)
  const [chatKey, setChatKey] = useState("initial");

  // ── Handlers ────────────────────────────────────────────────────────────

  const handleNewChat = useCallback(() => {
    createChat();                        // creates session + sets activeChatId
    setChatKey(crypto.randomUUID());     // force remount → blank window
  }, [createChat]);

  const handleSelectChat = useCallback(
    (id: string) => {
      selectChat(id);                    // sets activeChatId
      setChatKey(`select-${id}`);        // force remount → loads stored messages
    },
    [selectChat]
  );

  const handleDeleteChat = useCallback(
    (id: string) => {
      deleteChat(id);
      if (activeChatId === id) {
        // The active chat was deleted — clear the window
        setChatKey(crypto.randomUUID());
      }
    },
    [activeChatId, deleteChat]
  );

  // Called by ChatWindow the first time the user sends a message.
  // We use this to:
  //   1. Auto-create a chat session if none exists yet (welcome-screen flow)
  //   2. Set the chat title from the first message text
  //
  // Critically, we do NOT change chatKey here. The ChatWindow is already
  // mounted and holding the user's first message — remounting would lose it.
  // Instead, we just wire up the persisted session ID so future saves work.
  const handleFirstMessage = useCallback(
    (message: string) => {
      if (!activeChatId) {
        // User started typing on the welcome screen with no active chat.
        // Create the session and get its ID — chatKey stays the same so
        // ChatWindow does not remount (messages are preserved in memory).
        const newId = createChat();
        updateChatTitle(newId, message);
      } else {
        // Chat session already exists; just set its title.
        updateChatTitle(activeChatId, message);
      }
    },
    [activeChatId, createChat, updateChatTitle]
  );

  // ── SSR / hydration guard ───────────────────────────────────────────────
  // Don't render ChatWindow until localStorage has been read, otherwise
  // initialMessages would always be [] on the first render and a flash of
  // the empty state would be visible before the messages appear.
  if (!isHydrated) {
    return (
      <div className="flex h-dvh w-full items-center justify-center bg-white dark:bg-slate-950 transition-colors duration-500">
        <div className="flex flex-col items-center gap-3 opacity-60">
          <div className="h-10 w-10 rounded-xl bg-amber-100 dark:bg-amber-900/50 animate-pulse" />
          <p className="text-sm text-slate-400 dark:text-slate-500 font-medium">
            Loading AskAra+…
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-dvh w-full overflow-hidden overflow-x-hidden bg-white dark:bg-slate-950 transition-colors duration-500">
      {/* Sidebar */}
      <Sidebar
        chatHistory={chatHistory}
        activeChatId={activeChatId}
        onNewChat={handleNewChat}
        onSelectChat={handleSelectChat}
        onDeleteChat={handleDeleteChat}
      />

      {/* Main content */}
      <main className="flex-1 min-w-0 flex flex-col h-full relative overflow-hidden overflow-x-hidden bg-white dark:bg-slate-900 border-l border-slate-200 dark:border-slate-800 shadow-sm transition-colors duration-500">
        {/* Fixed Mobile Header */}
        <div className="flex-none h-16 md:hidden w-full border-b border-slate-100 dark:border-slate-800 flex items-center justify-between px-4 bg-white/90 dark:bg-slate-950/90 backdrop-blur-md z-40 transition-colors duration-500">
          <div className="w-10" />
          <h2 className="font-heading font-extrabold text-slate-900 dark:text-white text-lg tracking-tight">
            AskAra<span className="text-amber-600 dark:text-amber-400">+</span>
          </h2>
          <Button
            variant="ghost"
            size="icon"
            onClick={handleNewChat}
            className="text-amber-600 dark:text-amber-400 hover:bg-slate-50 dark:hover:bg-slate-900 rounded-xl transition-colors"
          >
            <MessageSquarePlus className="h-5 w-5" />
            <span className="sr-only">New Chat</span>
          </Button>
        </div>

        <div className="flex-1 overflow-hidden relative">
          {/*
           * key={chatKey}   — remounts when switching chats or creating new ones
           * chatId          — tells ChatWindow which localStorage slot to use
           * onFirstMessage  — auto-creates + titles the session on first send
           */}
          <ChatWindow
            key={chatKey}
            chatId={activeChatId ?? undefined}
            onFirstMessage={handleFirstMessage}
          />
        </div>
      </main>
    </div>
  );
}