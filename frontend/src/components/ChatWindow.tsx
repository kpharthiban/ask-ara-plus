"use client";

import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { ArrowUp, Square, WifiOff, Loader2, X, AlertCircle } from "lucide-react";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import ReactMarkdown from "react-markdown";
import MessageBubble from "./MessageBubble";
import StepCards from "./StepCards";
import RecommendationCards from "./RecommendationCards";
import TypingIndicator from "./TypingIndicator";
import LanguageSelector, { LANGUAGES, type LanguageOption } from "./LanguageSelector";
import VoiceInput from "./VoiceInput";
import VoiceOutput from "./VoiceOutput";
import AgentReasoning from "./AgentReasoning";
import ImageUpload, { type SelectedImage } from "./ImageUpload";
import ScenarioCards from "./ScenarioCards";
import ProfilingFlow from "./ProfilingFlow";
import LoadingSkeleton from "./LoadingSkeleton";
import { useWebSocket } from "@/hooks/useWebSocket";
import { loadChatMessages, saveChatMessages } from "@/hooks/usePersistentChats";
import type { Message, StepCardsPayload, RecommendationsPayload } from "@/lib/types";
import { cn } from "@/lib/utils";

// ── Response router ───────────────────────────────────────────────────────
// For structured messages (step_cards / recommendations):
//   Renders the streamed markdown text + cards together under ONE avatar.
//   The streamed text stays visible — cards appear below it smoothly.
// For text messages:
//   Renders normal MessageBubble.

function MessageRouter({ message }: { message: Message }) {
  // ── Structured: combined text + cards layout ──
  if (
    (message.contentType === "step_cards" || message.contentType === "recommendations") &&
    message.structured
  ) {
    return (
      <div className="flex w-full min-w-0 mb-8 justify-start">
        {/* AI avatar — matches MessageBubble */}
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-amber-100 dark:bg-amber-900/50 mr-4 mt-0.5 border border-amber-200/50 dark:border-amber-800/50 shadow-sm transition-colors duration-500 overflow-hidden">
          <Image src="/icons/cat.png" alt="Ara" width={20} height={20} className="object-contain" />
        </div>

        <div className="max-w-[85%] md:max-w-[75%] w-full min-w-0 overflow-hidden isolate">
          {/* Streamed text — rendered as markdown (same as MessageBubble) */}
          {message.content && message.content.trim() && (
            <div className="text-[16px] leading-relaxed text-slate-800 dark:text-slate-200 font-medium ara-markdown mb-4 break-words overflow-wrap-anywhere">
              <ReactMarkdown
                components={{
                  p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
                  strong: ({ children }) => (
                    <strong className="font-bold text-slate-900 dark:text-white">{children}</strong>
                  ),
                  em: ({ children }) => <em className="italic">{children}</em>,
                  a: ({ href, children }) => (
                    <a href={href} target="_blank" rel="noopener noreferrer"
                      className="text-amber-600 dark:text-amber-400 underline underline-offset-2 hover:text-amber-700 dark:hover:text-amber-300 transition-colors break-all">
                      {children}
                    </a>
                  ),
                  ul: ({ children }) => <ul className="mb-3 ml-1 space-y-1.5 last:mb-0">{children}</ul>,
                  ol: ({ children }) => <ol className="mb-3 ml-1 space-y-1.5 list-decimal list-inside last:mb-0">{children}</ol>,
                  li: ({ children }) => (
                    <li className="flex gap-2 items-start">
                      <span className="mt-2 h-1.5 w-1.5 rounded-full bg-amber-500 dark:bg-amber-400 shrink-0" />
                      <span className="flex-1">{children}</span>
                    </li>
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

          {/* Cards — rendered directly, no avatar/intro (handled above) */}
          {message.structured.type === "step_cards" && (
            <StepCards data={message.structured as StepCardsPayload} compact />
          )}
          {message.structured.type === "recommendations" && (
            <RecommendationCards data={message.structured as RecommendationsPayload} compact />
          )}

          {/* ── Action row: voice output ── */}
          <div className="flex items-center gap-2 mt-2 min-w-0">
            <VoiceOutput
              text={message.content}
              isStreaming={message.isStreaming}
            />
          </div>

          {/* ── Agent reasoning panel (collapsible) ── */}
          <AgentReasoning
            toolCalls={message.toolCalls}
            sources={message.sources}
            isStreaming={message.isStreaming}
          />
        </div>
      </div>
    );
  }

  // ── Text / other types: normal bubble ──
  return <MessageBubble message={message} />;
}

// ── ChatWindow ────────────────────────────────────────────────────────────

interface ChatWindowProps {
  /**
   * The active chat session ID, used as the localStorage key for this chat's
   * messages. When undefined (welcome/no-chat state), messages are held in
   * memory only and saved as soon as a chatId is assigned.
   */
  chatId?: string;
  /**
   * Called once with the first user message text so the parent can set the
   * chat title. Not called when restoring an existing chat (initialMessages
   * is non-empty).
   */
  onFirstMessage?: (message: string) => void;
}

export default function ChatWindow({ chatId, onFirstMessage }: ChatWindowProps = {}) {
  // ── Load persisted messages for this chat (only on mount) ──────────────
  // useMemo with an empty dep array runs exactly once per mount.
  // Since ChatWindow remounts when switching between existing chats
  // (key=chatKey in page.tsx), this is always correct.
  const initialMessages = useMemo(() => {
    if (!chatId) return [];
    return loadChatMessages(chatId);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // intentionally empty — chatId is stable for the lifetime of this mount

  const {
    messages: wsMessages,
    status,
    isThinking,
    activeTool,
    sendMessage,
    stopGeneration,
  } = useWebSocket(initialMessages);

  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // ── Language selection state ──
  const [selectedLang, setSelectedLang] = useState<LanguageOption>(LANGUAGES[0]); // "auto"

  // ── Toast state (for errors) ──
  const [toast, setToast] = useState<string | null>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── Pending image state ──
  const [pendingImage, setPendingImage] = useState<SelectedImage | null>(null);

  // ── Drag-and-drop state for welcome screen upload zone ──
  const [isDragOver, setIsDragOver] = useState(false);
  const dropZoneInputRef = useRef<HTMLInputElement | null>(null);

  const showToast = useCallback((message: string) => {
    setToast(message);
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 3500);
  }, []);

  const messages = wsMessages;

  const isConnected = status === "connected";
  const isStreaming = isThinking || wsMessages.some((m) => m.isStreaming);

  // ── Persist messages to localStorage after each completed exchange ──────
  // We deliberately skip writes while any message is still streaming/loading
  // to avoid storing partial content. Once isStreaming becomes false and all
  // messages are finalized, we write the full snapshot.
  useEffect(() => {
    // Nothing to save without a chat ID or messages
    if (!chatId || wsMessages.length === 0) return;

    // Don't write mid-stream
    const hasTransient = wsMessages.some((m) => m.isStreaming || m.isLoading);
    if (hasTransient) return;

    saveChatMessages(chatId, wsMessages);
  }, [chatId, wsMessages]);

  // Auto-scroll on new messages or while streaming
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isThinking]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Track if first message has been sent (for onFirstMessage callback).
  // If we loaded existing messages, the chat already has a title — skip.
  const firstMessageSentRef = useRef(initialMessages.length > 0);

  // ── Handle send ──
  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();

    const trimmed = input.trim();
    const hasImage = !!pendingImage;

    // Require at least text or image
    if (!trimmed && !hasImage) return;
    if (!isConnected) return;

    // Notify parent of first message (so it can set the chat title)
    if (!firstMessageSentRef.current && onFirstMessage) {
      firstMessageSentRef.current = true;
      onFirstMessage(trimmed || "📷 Image");
    }

    sendMessage(trimmed, {
      language: selectedLang.code || undefined,
      country: selectedLang.country || undefined,
      ...(hasImage && pendingImage
        ? {
            imageBase64: pendingImage.base64,
            imageMediaType: pendingImage.mediaType,
            imagePreview: pendingImage.previewUrl,
          }
        : {}),
    });

    setInput("");
    setPendingImage(null);
    inputRef.current?.focus();
  };

  // ── Scenario card tap → send as a regular message ──
  const handleScenarioSelect = useCallback(
    (prompt: string) => {
      if (!isConnected) return;

      // Treat scenario selection as first message for title purposes
      if (!firstMessageSentRef.current && onFirstMessage) {
        firstMessageSentRef.current = true;
        onFirstMessage(prompt);
      }

      sendMessage(prompt, {
        language: selectedLang.code || undefined,
        country: selectedLang.country || undefined,
      });
    },
    [isConnected, sendMessage, selectedLang, onFirstMessage]
  );

  // Show scenario cards only before user has sent any message
  const isEmptyChat = wsMessages.length === 0;

  // ── Proactive profiling flow state ──
  const [isProfiling, setIsProfiling] = useState(false);

  const handleStartProfiling = useCallback(() => {
    setIsProfiling(true);
  }, []);

  const handleProfilingComplete = useCallback(
    (message: string) => {
      setIsProfiling(false);
      if (!isConnected) return;

      if (!firstMessageSentRef.current && onFirstMessage) {
        firstMessageSentRef.current = true;
        onFirstMessage(message);
      }

      sendMessage(message, {
        language: selectedLang.code || undefined,
        country: selectedLang.country || undefined,
      });
    },
    [isConnected, sendMessage, selectedLang, onFirstMessage]
  );

  const handleProfilingCancel = useCallback(() => {
    setIsProfiling(false);
  }, []);

  // ── Image upload handler ──
  const handleImageSelected = useCallback((image: SelectedImage) => {
    setPendingImage(image);
    // Focus back to input so user can type a question
    inputRef.current?.focus();
  }, []);

  // ── Remove pending image ──
  const handleRemoveImage = useCallback(() => {
    setPendingImage(null);
  }, []);

  // ── Shared file → SelectedImage processing (used by drop zone + input) ──
  const processImageFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      showToast("Please select an image file.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showToast("Image is too large. Please choose one under 10 MB.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      const [prefix, base64] = dataUrl.split(",");
      const mediaType = prefix.replace("data:", "").replace(";base64", "");
      handleImageSelected({ base64, previewUrl: dataUrl, mediaType });
    };
    reader.onerror = () => showToast("Could not read the image. Please try again.");
    reader.readAsDataURL(file);
  }, [handleImageSelected, showToast]);

  // ── Drop zone drag handlers ──
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) processImageFile(file);
  }, [processImageFile]);

  const handleDropZoneClick = useCallback(() => {
    dropZoneInputRef.current?.click();
  }, []);

  const handleDropZoneFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processImageFile(file);
    e.target.value = "";
  }, [processImageFile]);

  // Whether to show the send button (text OR image pending)
  const canSend = !!input.trim() || !!pendingImage;

  return (
    <div className="flex flex-col flex-1 min-w-0 h-full w-full max-w-full bg-slate-50/50 dark:bg-slate-900/50 bg-dot-pattern relative transition-colors duration-500 overflow-hidden overflow-x-hidden">

      {/* Scrollable Messages Area */}
      <div className="flex-1 w-full overflow-x-hidden overflow-y-hidden relative">
        {/* Scroll fade — top */}
        <div className="absolute top-0 left-0 right-0 h-6 bg-gradient-to-b from-slate-50/80 dark:from-slate-900/80 to-transparent z-10 pointer-events-none" />

        <div className="h-full w-full overflow-y-auto overflow-x-hidden">
          <div className="flex flex-col pt-8 pb-8 w-full max-w-3xl mx-auto min-w-0 px-4 md:px-8 overflow-x-hidden">

            {/* ── Branded welcome hero — empty chat state ── */}
            {isEmptyChat && !isProfiling && (
              <div className="flex flex-col items-center text-center mb-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-amber-100 dark:bg-amber-900/50 mb-4 border border-amber-200/50 dark:border-amber-800/50 shadow-sm overflow-hidden">
                  <Image src="/icons/cat.png" alt="Ara" width={36} height={36} className="object-contain" />
                </div>
                <h2 className="font-heading text-xl font-extrabold text-slate-900 dark:text-white tracking-tight">
                  Hi, I&apos;m Ara <span className="inline-block animate-in fade-in duration-700 delay-300">✨</span>
                </h2>
                <p className="text-sm text-slate-500 dark:text-slate-400 mt-1.5 max-w-xs leading-relaxed">
                  Your multilingual guide to Malaysian government services for migrants, workers &amp; small businesses. (to be expanded across ASEAN and more services)
                </p>
                {/* Malaysia knowledge base badge */}
                <div className="mt-3 inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
                  <span className="text-sm">🇲🇾</span>
                  <span className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 tracking-wide uppercase">Malaysia Knowledge Base</span>
                </div>
              </div>
            )}

            {/* Messages */}
            {messages.map((msg) => (
              <MessageRouter key={msg.id} message={msg} />
            ))}

            {/* Scenario quick-access cards — empty chat state */}
            {isEmptyChat && !isProfiling && (
              <>
                {/* ── Drag & Drop / Click-to-Upload zone ── */}
                <div className="w-full max-w-md mx-auto mb-3 animate-in fade-in slide-in-from-bottom-4 duration-500 delay-100">
                  {/* Hidden file input for the drop zone */}
                  <input
                    ref={dropZoneInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleDropZoneFileChange}
                    aria-hidden="true"
                    tabIndex={-1}
                  />
                  <button
                    type="button"
                    onClick={handleDropZoneClick}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    disabled={!isConnected}
                    className={cn(
                      "w-full flex flex-col items-center justify-center gap-2 px-4 py-5 rounded-2xl border-2 border-dashed transition-all duration-200 cursor-pointer text-center",
                      isDragOver
                        ? "border-amber-400 dark:border-amber-500 bg-amber-50 dark:bg-amber-900/20 scale-[1.01]"
                        : "border-slate-200 dark:border-slate-700 bg-white/60 dark:bg-slate-800/40 hover:border-amber-300 dark:hover:border-amber-700 hover:bg-amber-50/40 dark:hover:bg-amber-900/10",
                      "disabled:opacity-40 disabled:cursor-not-allowed"
                    )}
                  >
                    <div className={cn(
                      "flex h-9 w-9 items-center justify-center rounded-xl transition-colors duration-200",
                      isDragOver
                        ? "bg-amber-100 dark:bg-amber-800/50"
                        : "bg-slate-100 dark:bg-slate-700/60"
                    )}>
                      <svg
                        className={cn("h-5 w-5 transition-colors duration-200", isDragOver ? "text-amber-500" : "text-slate-400 dark:text-slate-500")}
                        fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                      </svg>
                    </div>
                    <div>
                      <p className={cn("text-sm font-semibold transition-colors duration-200", isDragOver ? "text-amber-600 dark:text-amber-400" : "text-slate-600 dark:text-slate-300")}>
                        {isDragOver ? "Drop image here" : "Upload a document photo"}
                      </p>
                      <p className="text-[11px] text-slate-400 dark:text-slate-500 mt-0.5">
                        Drag &amp; drop or click — letter, permit, form, contract
                      </p>
                    </div>
                  </button>
                </div>

                <ScenarioCards
                  onSelect={handleScenarioSelect}
                  onStartProfiling={handleStartProfiling}
                  disabled={!isConnected}
                />
              </>
            )}

            {/* Proactive profiling quick-reply flow */}
            {isProfiling && (
              <ProfilingFlow
                onComplete={handleProfilingComplete}
                onCancel={handleProfilingCancel}
                disabled={!isConnected}
              />
            )}

            {/* Loading skeleton */}
            {isThinking && !activeTool && messages.length > 0 && !messages[messages.length - 1]?.isStreaming && (
              <LoadingSkeleton />
            )}

            {/* Typing / tool-use indicator */}
            {isThinking && activeTool && <TypingIndicator activeTool={activeTool} />}

            {/* Auto-scroll anchor */}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Scroll fade — bottom */}
        <div className="absolute bottom-0 left-0 right-0 h-6 bg-gradient-to-t from-slate-50/80 dark:from-slate-900/80 to-transparent z-10 pointer-events-none" />
      </div>

      {/* Sticky Bottom Input Area */}
      <div className="w-full min-w-0 bg-white dark:bg-slate-950 border-t border-slate-200 dark:border-slate-800 shrink-0 transition-colors duration-500 overflow-hidden">
        <div className="max-w-3xl mx-auto w-full p-4 md:p-6 pb-safe">

          {/* Connection banner */}
          {!isConnected && (
            <div className="flex items-center justify-center gap-2 mb-3 text-xs">
              {status === "connecting" ? (
                <>
                  <Loader2 className="h-3 w-3 animate-spin text-amber-500" />
                  <span className="text-amber-600 dark:text-amber-400">
                    Reconnecting to Ara...
                  </span>
                </>
              ) : (
                <>
                  <WifiOff className="h-3 w-3 text-red-500" />
                  <span className="text-red-600 dark:text-red-400">
                    Disconnected — check your connection
                  </span>
                </>
              )}
            </div>
          )}

          {/* ── Error toast ── */}
          {toast && (
            <div className="flex items-center justify-center gap-2 mb-3 px-4 py-2.5 rounded-xl bg-red-50 dark:bg-red-950/50 border border-red-200 dark:border-red-800 animate-in fade-in slide-in-from-bottom-2 duration-200">
              <AlertCircle className="h-4 w-4 text-red-500 dark:text-red-400 shrink-0" />
              <span className="text-sm font-medium text-red-700 dark:text-red-300">
                {toast}
              </span>
              <button
                onClick={() => setToast(null)}
                className="ml-auto p-0.5 rounded-md text-red-400 hover:text-red-600 dark:hover:text-red-200 transition-colors"
                aria-label="Dismiss"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          )}

          {/* ── Language Selector — above input bar ── */}
          <div className="mb-2 w-full overflow-hidden">
            <LanguageSelector
              selected={selectedLang.id}
              onChange={setSelectedLang}
            />
          </div>

          {/* ── Image preview strip — appears above the input box when image is pending ── */}
          {pendingImage && (
            <div className="mb-2 flex items-start gap-2 animate-in fade-in slide-in-from-bottom-2 duration-200">
              <div className="relative group">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={pendingImage.previewUrl}
                  alt="Attached image"
                  className="h-20 w-20 object-cover rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm"
                />
                {/* Remove button */}
                <button
                  type="button"
                  onClick={handleRemoveImage}
                  className="absolute -top-2 -right-2 h-5 w-5 rounded-full bg-slate-800 dark:bg-slate-200 text-white dark:text-slate-900 flex items-center justify-center shadow-md opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity"
                  aria-label="Remove image"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
              <p className="text-xs text-slate-400 dark:text-slate-500 mt-1 leading-relaxed">
                Image attached.<br />Type a question or send as-is.
              </p>
            </div>
          )}

          <form
            onSubmit={handleSend}
            className="flex items-center gap-2 p-1.5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-2xl focus-within:border-amber-500 dark:focus-within:border-amber-400 focus-within:ring-1 focus-within:ring-amber-500 transition-all shadow-sm"
          >
            {/* Image Upload Button */}
            <ImageUpload
              onImageSelected={handleImageSelected}
              onError={showToast}
              disabled={!isConnected}
            />

            {/* Input Field */}
            <Input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                pendingImage
                  ? "Ask about this image… (optional)"
                  : isConnected
                  ? "Message Ara..."
                  : "Connecting..."
              }
              disabled={!isConnected}
              className="flex-1 h-12 border-0 bg-transparent px-2 text-[16px] text-slate-900 dark:text-white placeholder:text-slate-400 focus-visible:ring-0 shadow-none rounded-none disabled:opacity-50"
            />

            {/* Action Buttons */}
            {isStreaming ? (
              <Button
                type="button"
                size="icon"
                onClick={stopGeneration}
                className="rounded-xl h-12 w-12 shrink-0 bg-amber-600 dark:bg-amber-500 hover:bg-amber-700 dark:hover:bg-amber-400 text-white dark:text-slate-950 transition-transform active:scale-95 shadow-sm"
              >
                <Square className="h-4 w-4 fill-current" />
              </Button>
            ) : canSend ? (
              <Button
                type="submit"
                size="icon"
                disabled={!isConnected}
                className="rounded-xl h-12 w-12 shrink-0 bg-amber-600 dark:bg-amber-500 hover:bg-amber-700 dark:hover:bg-amber-400 text-white dark:text-slate-950 transition-transform active:scale-95 shadow-sm disabled:opacity-50"
              >
                <ArrowUp className="h-5 w-5" />
              </Button>
            ) : (
              <VoiceInput
                langCode={selectedLang.code}
                onTranscript={(text) => setInput((prev) => (prev ? prev + " " + text : text))}
                onError={showToast}
                disabled={!isConnected}
              />
            )}
          </form>

          <div className="text-center mt-3">
            <p className="text-[12px] text-slate-400 dark:text-slate-500 font-medium">
              Ara can make mistakes. Verify critical government information.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}