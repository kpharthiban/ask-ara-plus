"use client";

import { useState, useEffect } from "react";
import { Plus, Menu, Sun, Moon, MessageSquare, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger, SheetTitle } from "@/components/ui/sheet";
import { useTheme } from "next-themes";
import Image from "next/image";
import { cn } from "@/lib/utils";

// ── Chat session type ─────────────────────────────────────────────────────
export interface ChatSession {
  id: string;
  title: string;
  createdAt: Date;
  /** Updated whenever a new message is saved — used for sorting & display. */
  updatedAt?: Date;
}

// ── Relative date formatter ───────────────────────────────────────────────
function formatRelativeDate(date: Date | undefined): string {
  if (!date) return "";
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / (1000 * 60));
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

// ── Props ─────────────────────────────────────────────────────────────────
interface SidebarProps {
  chatHistory: ChatSession[];
  activeChatId: string | null;
  onNewChat: () => void;
  onSelectChat: (id: string) => void;
  onDeleteChat?: (id: string) => void;
}

export default function Sidebar({
  chatHistory,
  activeChatId,
  onNewChat,
  onSelectChat,
  onDeleteChat,
}: SidebarProps) {
  const [isOpen, setIsOpen] = useState(false);
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // Prevent hydration mismatch for the theme toggle
  useEffect(() => setMounted(true), []);

  const SidebarContent = () => (
    <div className="flex h-full flex-col p-4 bg-white dark:bg-slate-950 border-r border-slate-100 dark:border-slate-800 transition-colors duration-500">

      {/* Brand Header with Cat Logo */}
      <div className="flex items-center gap-3 px-2 py-4 mb-6 animate-in slide-in-from-left-4 fade-in duration-500">
        <div className="flex h-10 w-10 items-center justify-center bg-amber-100 dark:bg-amber-500/20 rounded-xl shadow-sm border border-amber-200 dark:border-amber-500/30 overflow-hidden">
          <Image src="/icons/cat.png" alt="Ara" width={24} height={24} className="object-contain" />
        </div>
        <div>
          <h1 className="font-heading text-xl font-extrabold text-slate-900 dark:text-white tracking-tight leading-none">
            AskAra<span className="text-amber-600 dark:text-amber-400">+</span>
          </h1>
          <p className="text-[10px] text-slate-500 dark:text-slate-400 font-semibold tracking-wide uppercase mt-1 leading-snug">
            Your companion for<br/>public services across ASEAN
          </p>
        </div>
      </div>

      {/* New Chat Button */}
      <Button
        onClick={() => {
          onNewChat();
          setIsOpen(false);
        }}
        className="w-full justify-start gap-3 bg-slate-900 dark:bg-white hover:bg-slate-800 dark:hover:bg-slate-200 text-white dark:text-slate-900 rounded-xl mb-6 py-6 h-auto transition-transform active:scale-95 shadow-sm animate-in slide-in-from-left-4 fade-in duration-500 delay-75"
      >
        <Plus className="h-5 w-5" />
        <span className="font-bold text-[15px]">New Chat</span>
      </Button>

      {/* Chat History */}
      <div className="flex-1 overflow-y-auto scrollbar-none space-y-1 animate-in slide-in-from-left-4 fade-in duration-500 delay-150">
        <p className="text-[10px] font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider px-2 mb-2">
          Recent Chats
        </p>

        {chatHistory.length === 0 && (
          <p className="text-xs text-slate-400 dark:text-slate-500 px-2 py-4 text-center">
            No conversations yet
          </p>
        )}

        {chatHistory.map((chat) => {
          const displayDate = formatRelativeDate(chat.updatedAt ?? chat.createdAt);
          const isActive = activeChatId === chat.id;

          return (
            <div
              key={chat.id}
              className={cn(
                "group flex items-start gap-2.5 px-3 py-2.5 rounded-xl cursor-pointer transition-colors",
                isActive
                  ? "bg-amber-50 dark:bg-amber-900/20 border border-amber-200/60 dark:border-amber-800/40"
                  : "hover:bg-slate-50 dark:hover:bg-slate-900"
              )}
              onClick={() => {
                onSelectChat(chat.id);
                setIsOpen(false);
              }}
            >
              <MessageSquare
                className={cn(
                  "h-4 w-4 shrink-0 mt-0.5",
                  isActive
                    ? "text-amber-600 dark:text-amber-400"
                    : "text-slate-400 dark:text-slate-500"
                )}
              />

              <div className="flex-1 min-w-0">
                <span
                  className={cn(
                    "block text-sm font-medium truncate leading-snug",
                    isActive
                      ? "text-slate-900 dark:text-white"
                      : "text-slate-600 dark:text-slate-400"
                  )}
                >
                  {chat.title}
                </span>

                {displayDate && (
                  <span className="block text-[10px] text-slate-400 dark:text-slate-600 mt-0.5 leading-none">
                    {displayDate}
                  </span>
                )}
              </div>

              {onDeleteChat && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteChat(chat.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 rounded-md text-slate-400 hover:text-red-500 dark:hover:text-red-400 transition-all shrink-0 mt-0.5"
                  aria-label="Delete chat"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
          );
        })}
      </div>

      {/* Theme Toggle at Bottom */}
      {mounted && (
        <div className="mt-auto pt-4 border-t border-slate-100 dark:border-slate-800 animate-in fade-in duration-500 delay-300">
          <button
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="w-full flex items-center justify-between px-4 py-3 text-[13px] font-medium text-slate-600 dark:text-slate-400 bg-slate-50 dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-xl transition-colors"
          >
            <span>Appearance</span>
            {theme === "dark" ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
          </button>
        </div>
      )}
    </div>
  );

  return (
    <>
      {/* Mobile Menu Trigger */}
      <div className="md:hidden fixed top-3 left-4 z-50">
        <Sheet open={isOpen} onOpenChange={setIsOpen}>
          <SheetTrigger asChild>
            <Button
              variant="outline"
              size="icon"
              className="rounded-xl bg-white dark:bg-slate-950 border-slate-200 dark:border-slate-800 text-slate-900 dark:text-white shadow-sm transition-colors duration-500"
            >
              <Menu className="h-5 w-5" />
            </Button>
          </SheetTrigger>
          <SheetContent side="left" className="p-0 w-[280px] border-none bg-white dark:bg-slate-950">
            <SheetTitle className="sr-only">Menu</SheetTitle>
            <SidebarContent />
          </SheetContent>
        </Sheet>
      </div>

      {/* Desktop Sidebar */}
      <aside className="hidden md:flex w-[280px] flex-col h-dvh z-10 shrink-0">
        <SidebarContent />
      </aside>
    </>
  );
}