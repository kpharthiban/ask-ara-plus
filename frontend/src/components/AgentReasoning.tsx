"use client";

import { useState } from "react";
import { ChevronDown, Brain } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Source } from "@/lib/types";

// ── Tool → friendly display mapping ─────────────────────────────────────────

interface ToolDisplay {
  icon: string;
  label: string;
  description: string;
}

const TOOL_MAP: Record<string, ToolDisplay> = {
  search_documents: {
    icon: "📚",
    label: "Knowledge Search",
    description: "Searched government document database",
  },
  detect_language: {
    icon: "🔍",
    label: "Language Detection",
    description: "Detected input language and dialect",
  },
  translate: {
    icon: "🌐",
    label: "Translation",
    description: "Translated content across languages",
  },
  simplify: {
    icon: "✏️",
    label: "Simplification",
    description: "Simplified to Grade 5 reading level",
  },
  summarize: {
    icon: "📋",
    label: "Summarization",
    description: "Created structured step-by-step guide",
  },
  dialect_adapt: {
    icon: "🗣️",
    label: "Dialect Adaptation",
    description: "Adapted to regional dialect",
  },
  assess_complexity: {
    icon: "📊",
    label: "Complexity Assessment",
    description: "Assessed text reading difficulty",
  },
  scan_document: {
    icon: "📷",
    label: "Document Scan",
    description: "Extracted text from photographed document",
  },
  fetch_gov_portal: {
    icon: "🌐",
    label: "Live Portal Fetch",
    description: "Retrieved fresh data from government portal",
  },
  profile_match: {
    icon: "🎯",
    label: "Profile Matching",
    description: "Matched profile against eligible programs",
  },
  text_to_speech: {
    icon: "🔊",
    label: "Text-to-Speech",
    description: "Converted response to audio",
  },
  // Fallbacks
  fallback_direct_llm: {
    icon: "⚡",
    label: "Direct Response",
    description: "Responded directly without tool chain",
  },
};

function getToolDisplay(toolName: string): ToolDisplay {
  // Try exact match first
  if (TOOL_MAP[toolName]) return TOOL_MAP[toolName];

  // Try matching by substring (tools might come with prefixes)
  const key = Object.keys(TOOL_MAP).find((k) =>
    toolName.toLowerCase().includes(k.toLowerCase())
  );
  if (key) return TOOL_MAP[key];

  // Fallback: humanize the tool name
  return {
    icon: "🔧",
    label: toolName
      .replace(/_/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase()),
    description: `Used ${toolName}`,
  };
}

// ── Props ────────────────────────────────────────────────────────────────────

interface AgentReasoningProps {
  toolCalls?: string[];
  sources?: Source[];
  isStreaming?: boolean;
}

// ── Component ────────────────────────────────────────────────────────────────

export default function AgentReasoning({ toolCalls, sources, isStreaming }: AgentReasoningProps) {
  const [isOpen, setIsOpen] = useState(false);

  // Don't render while streaming or if no reasoning data
  const hasTools = toolCalls && toolCalls.length > 0;
  const hasSources = sources && sources.length > 0;

  if (isStreaming || (!hasTools && !hasSources)) return null;

  return (
    <div className="mt-2 w-full min-w-0">
      {/* Toggle button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-all duration-200",
          "text-slate-400 dark:text-slate-500",
          "hover:text-slate-600 dark:hover:text-slate-300",
          "hover:bg-slate-100 dark:hover:bg-slate-800",
          isOpen && "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300"
        )}
      >
        <Brain className="h-3 w-3" />
        <span>How Ara found this</span>
        <ChevronDown
          className={cn(
            "h-3 w-3 transition-transform duration-200",
            isOpen && "rotate-180"
          )}
        />
      </button>

      {/* Collapsible panel */}
      <div
        className={cn(
          "overflow-hidden transition-all duration-300 ease-in-out",
          isOpen ? "max-h-96 opacity-100 mt-2" : "max-h-0 opacity-0"
        )}
      >
        <div className="px-3 py-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200/50 dark:border-slate-700/50 space-y-2 overflow-hidden">

          {/* Tool chain */}
          {hasTools && (
            <div className="space-y-1.5">
              <p className="text-[10px] font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider">
                Reasoning Chain
              </p>
              <div className="space-y-1">
                {toolCalls!.map((tool, i) => {
                  const display = getToolDisplay(tool);
                  return (
                    <div
                      key={i}
                      className="flex items-start gap-2 text-xs text-slate-600 dark:text-slate-300"
                    >
                      {/* Step number + connector line */}
                      <div className="flex flex-col items-center shrink-0">
                        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-amber-100 dark:bg-amber-900/40 text-[10px] font-bold text-amber-700 dark:text-amber-400">
                          {i + 1}
                        </span>
                        {i < toolCalls!.length - 1 && (
                          <div className="w-px h-2 bg-slate-200 dark:bg-slate-700 mt-0.5" />
                        )}
                      </div>

                      {/* Tool info */}
                      <div className="flex items-center gap-1.5 min-h-[20px] min-w-0 flex-wrap">
                        <span className="text-sm leading-none">{display.icon}</span>
                        <span className="font-medium">{display.label}</span>
                        <span className="text-slate-400 dark:text-slate-500">—</span>
                        <span className="text-slate-500 dark:text-slate-400">
                          {display.description}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Sources (if present) */}
          {hasSources && (
            <div className="space-y-1.5">
              {hasTools && (
                <hr className="border-slate-200 dark:border-slate-700" />
              )}
              <p className="text-[10px] font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider">
                Sources Referenced
              </p>
              <div className="space-y-1">
                {sources!.map((source, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400 min-w-0"
                  >
                    <span className="text-sm leading-none">📄</span>
                    {source.url ? (
                      <a
                        href={
                          source.url.startsWith("http://") || source.url.startsWith("https://")
                            ? source.url
                            : `https://${source.url}`
                        }
                        target="_blank"
                        rel="noopener noreferrer"
                        className="hover:text-amber-600 dark:hover:text-amber-400 underline underline-offset-2 truncate transition-colors"
                      >
                        {source.title}
                      </a>
                    ) : (
                      <span className="truncate">{source.title}</span>
                    )}
                    {source.country && (
                      <span className="text-slate-300 dark:text-slate-600 shrink-0">
                        ({source.country})
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}