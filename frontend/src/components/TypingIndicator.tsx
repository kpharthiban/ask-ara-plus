/**
 * TypingIndicator.tsx
 * Shows "Ara is thinking..." with animated dots,
 * and which MCP tool is active if applicable.
 */

import { Search, Languages, FileText, Globe, Loader2 } from "lucide-react";
import Image from "next/image";

const TOOL_CONFIG: Record<string, { label: string; icon: typeof Search }> = {
  search_documents:   { label: "Searching documents",        icon: Search },
  detect_language:    { label: "Detecting language",          icon: Languages },
  simplify:           { label: "Simplifying",                 icon: FileText },
  translate:          { label: "Translating",                 icon: Languages },
  summarize:          { label: "Summarizing",                 icon: FileText },
  dialect_adapt:      { label: "Adapting dialect",            icon: Languages },
  assess_complexity:  { label: "Checking readability",        icon: FileText },
  scan_document:      { label: "Reading document",            icon: FileText },
  fetch_gov_portal:   { label: "Checking government portal",  icon: Globe },
  profile_match:      { label: "Matching programs",           icon: Search },
  text_to_speech:     { label: "Preparing audio",             icon: FileText },
};

interface TypingIndicatorProps {
  activeTool?: string | null;
}

export default function TypingIndicator({ activeTool }: TypingIndicatorProps) {
  const toolInfo = activeTool ? TOOL_CONFIG[activeTool] : null;
  const label = toolInfo?.label || (activeTool ? `Using ${activeTool}` : "Thinking");
  const ToolIcon = toolInfo?.icon || Loader2;

  return (
    <div className="flex w-full min-w-0 mb-8 justify-start">
      {/* Avatar — same as MessageBubble */}
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-amber-100 dark:bg-amber-900/50 mr-4 mt-0.5 border border-amber-200/50 dark:border-amber-800/50 shadow-sm transition-colors duration-500 overflow-hidden">
        <Image src="/icons/cat.png" alt="Ara" width={20} height={20} className="object-contain" />
      </div>

      {/* Indicator content */}
      <div className="flex items-center gap-2.5 py-1">
        {/* Tool icon */}
        {activeTool ? (
          <ToolIcon className="h-4 w-4 text-amber-600 dark:text-amber-400 animate-pulse" />
        ) : null}

        {/* Label */}
        <span className="text-sm text-slate-400 dark:text-slate-500 font-medium">
          {label}
        </span>

        {/* Animated dots */}
        <span className="inline-flex items-center gap-0.5">
          <span
            className="h-1.5 w-1.5 rounded-full bg-amber-500/60 dark:bg-amber-400/60 animate-bounce"
            style={{ animationDelay: "0ms", animationDuration: "0.8s" }}
          />
          <span
            className="h-1.5 w-1.5 rounded-full bg-amber-500/60 dark:bg-amber-400/60 animate-bounce"
            style={{ animationDelay: "150ms", animationDuration: "0.8s" }}
          />
          <span
            className="h-1.5 w-1.5 rounded-full bg-amber-500/60 dark:bg-amber-400/60 animate-bounce"
            style={{ animationDelay: "300ms", animationDuration: "0.8s" }}
          />
        </span>
      </div>
    </div>
  );
}
