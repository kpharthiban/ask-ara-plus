import { Message } from "@/lib/types";
import { cn } from "@/lib/utils";
import Image from "next/image";
import ReactMarkdown from "react-markdown";
import VoiceOutput from "./VoiceOutput";
import AgentReasoning from "./AgentReasoning";

export default function MessageBubble({ message }: { message: Message }) {
  const isUser = message.sender === "user";

  return (
    <div className={cn("flex w-full min-w-0 mb-8", isUser ? "justify-end" : "justify-start")}>

      {/* Premium AI Avatar */}
      {!isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-amber-100 dark:bg-amber-900/50 mr-4 mt-0.5 border border-amber-200/50 dark:border-amber-800/50 shadow-sm transition-colors duration-500 overflow-hidden">
          <Image src="/icons/cat.png" alt="Ara" width={20} height={20} className="object-contain" />
        </div>
      )}

      <div
        className={cn(
          "relative max-w-[85%] md:max-w-[75%] min-w-0 overflow-hidden text-[16px] leading-relaxed transition-colors duration-500",
          isUser
            ? "bg-slate-50 dark:bg-slate-800 text-slate-900 dark:text-white px-5 py-3.5 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm"
            : "bg-transparent text-slate-800 dark:text-slate-200 px-0 py-1"
        )}
      >
        {/* ── Image thumbnail (user messages only) ── */}
        {isUser && message.imagePreview && (
          <div className="mb-2.5 -mx-1">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={message.imagePreview}
              alt="Attached image"
              className="max-h-52 max-w-full rounded-xl object-contain border border-slate-200 dark:border-slate-600 shadow-sm"
            />
          </div>
        )}

        {/* Message content */}
        {message.contentType === "text" && (
          <div className="relative">
            {isUser ? (
              /* User messages: plain text, no markdown */
              message.content ? (
                <p className="whitespace-pre-wrap break-words overflow-wrap-anywhere font-medium">
                  {message.content}
                </p>
              ) : null
            ) : (
              /* Ara messages: render markdown */
              <div className="ara-markdown font-medium break-words overflow-wrap-anywhere">
                <ReactMarkdown
                  components={{
                    /* Paragraphs */
                    p: ({ children }) => (
                      <p className="mb-3 last:mb-0">{children}</p>
                    ),
                    /* Bold */
                    strong: ({ children }) => (
                      <strong className="font-bold text-slate-900 dark:text-white">
                        {children}
                      </strong>
                    ),
                    /* Italic */
                    em: ({ children }) => (
                      <em className="italic">{children}</em>
                    ),
                    /* Links */
                    a: ({ href, children }) => (
                      <a
                        href={href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-amber-600 dark:text-amber-400 underline underline-offset-2 hover:text-amber-700 dark:hover:text-amber-300 transition-colors break-all"
                      >
                        {children}
                      </a>
                    ),
                    /* Unordered lists */
                    ul: ({ children }) => (
                      <ul className="mb-3 ml-1 space-y-1.5 last:mb-0">{children}</ul>
                    ),
                    /* Ordered lists */
                    ol: ({ children }) => (
                      <ol className="mb-3 ml-1 space-y-1.5 list-decimal list-inside last:mb-0">
                        {children}
                      </ol>
                    ),
                    /* List items — custom bullet for unordered */
                    li: ({ children }) => (
                      <li className="flex gap-2 items-start">
                        <span className="mt-2 h-1.5 w-1.5 rounded-full bg-amber-500 dark:bg-amber-400 shrink-0" />
                        <span className="flex-1">{children}</span>
                      </li>
                    ),
                    /* Headings */
                    h1: ({ children }) => (
                      <h1 className="text-lg font-bold text-slate-900 dark:text-white mb-2">
                        {children}
                      </h1>
                    ),
                    h2: ({ children }) => (
                      <h2 className="text-base font-bold text-slate-900 dark:text-white mb-2">
                        {children}
                      </h2>
                    ),
                    h3: ({ children }) => (
                      <h3 className="text-sm font-bold text-slate-900 dark:text-white mb-1.5">
                        {children}
                      </h3>
                    ),
                    /* Code inline / block */
                    code: ({ children, className }) => {
                      const isBlock = className?.includes("language-");
                      if (isBlock) {
                        return (
                          <code className="block bg-slate-100 dark:bg-slate-800 rounded-lg px-4 py-3 text-sm font-mono overflow-x-auto mb-3">
                            {children}
                          </code>
                        );
                      }
                      return (
                        <code className="bg-slate-100 dark:bg-slate-800 rounded px-1.5 py-0.5 text-sm font-mono">
                          {children}
                        </code>
                      );
                    },
                    /* Code blocks */
                    pre: ({ children }) => (
                      <pre className="mb-3 last:mb-0">{children}</pre>
                    ),
                    /* Blockquotes */
                    blockquote: ({ children }) => (
                      <blockquote className="border-l-2 border-amber-500 dark:border-amber-400 pl-4 mb-3 text-slate-600 dark:text-slate-300 italic last:mb-0">
                        {children}
                      </blockquote>
                    ),
                    /* Horizontal rule */
                    hr: () => (
                      <hr className="border-slate-200 dark:border-slate-700 my-3" />
                    ),
                  }}
                >
                  {message.content}
                </ReactMarkdown>

                {/* Streaming cursor — blinking bar while tokens arrive */}
                {message.isStreaming && (
                  <span className="inline-block w-0.5 h-[1.1em] bg-amber-500 dark:bg-amber-400 ml-0.5 animate-pulse align-text-bottom" />
                )}
              </div>
            )}
          </div>
        )}

        {/* ── Action row: voice output ── */}
        {!isUser && (
          <div className="flex items-center gap-2 mt-2 min-w-0">
            <VoiceOutput
              text={message.content}
              isStreaming={message.isStreaming}
            />
          </div>
        )}

        {/* ── Agent reasoning panel (collapsible) ── */}
        {!isUser && (
          <AgentReasoning
            toolCalls={message.toolCalls}
            sources={message.sources}
            isStreaming={message.isStreaming}
          />
        )}
      </div>
    </div>
  );
}