"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Volume2, VolumeX, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

// ── Config ───────────────────────────────────────────────────────────────────

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Strip markdown syntax so TTS reads clean text */
function stripMarkdown(md: string): string {
  return md
    .replace(/#{1,6}\s+/g, "")           // headings
    .replace(/\*\*(.+?)\*\*/g, "$1")     // bold
    .replace(/\*(.+?)\*/g, "$1")         // italic
    .replace(/__(.+?)__/g, "$1")         // bold alt
    .replace(/_(.+?)_/g, "$1")           // italic alt
    .replace(/~~(.+?)~~/g, "$1")         // strikethrough
    .replace(/`{1,3}[^`]*`{1,3}/g, "")  // inline/block code
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1") // links → text only
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, "$1") // images → alt text
    .replace(/^\s*[-*+]\s+/gm, "")       // unordered list bullets
    .replace(/^\s*\d+\.\s+/gm, "")       // ordered list numbers
    .replace(/^\s*>\s+/gm, "")           // blockquotes
    .replace(/---+/g, "")                // horizontal rules
    .replace(/\n{2,}/g, ". ")            // double newlines → pause
    .replace(/\n/g, " ")                 // single newlines → space
    .trim();
}

/**
 * Browser speechSynthesis fallback — prefer female voice.
 */
function pickVoice(langTag: string): SpeechSynthesisVoice | null {
  const voices = speechSynthesis.getVoices();
  if (!voices.length) return null;

  const isFemale = (v: SpeechSynthesisVoice) =>
    /female|woman|zira|samantha|karen|moira|fiona|tessa|google.*female/i.test(v.name);

  const preferFemale = (candidates: SpeechSynthesisVoice[]) =>
    candidates.find(isFemale) || candidates[0] || null;

  if (!langTag) return preferFemale(voices);

  const exactMatches = voices.filter(
    (v) => v.lang.toLowerCase() === langTag.toLowerCase()
  );
  if (exactMatches.length) return preferFemale(exactMatches);

  const base = langTag.split("-")[0].toLowerCase();
  const partialMatches = voices.filter((v) =>
    v.lang.toLowerCase().startsWith(base)
  );
  if (partialMatches.length) return preferFemale(partialMatches);

  return null;
}

// ── Props ────────────────────────────────────────────────────────────────────

interface VoiceOutputProps {
  /** The raw message content (may contain markdown) */
  text: string;
  /** BCP-47 language tag for voice selection (e.g. "ms-MY", "th-TH") */
  langCode?: string;
  /** Don't show the button while still streaming */
  isStreaming?: boolean;
}

// ── Component ────────────────────────────────────────────────────────────────

export default function VoiceOutput({ text, langCode, isStreaming }: VoiceOutputProps) {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Refs for playback control
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Track which engine is active: "edge" | "browser" | null
  const activeEngine = useRef<"edge" | "browser" | null>(null);

  // ── Cleanup on unmount ──
  useEffect(() => {
    return () => {
      stopAll();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** Stop whatever is currently playing */
  const stopAll = useCallback(() => {
    // Stop edge-tts audio
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.removeAttribute("src");
      audioRef.current = null;
    }
    // Abort in-flight fetch
    abortRef.current?.abort();
    abortRef.current = null;
    // Stop browser speechSynthesis
    if (typeof speechSynthesis !== "undefined") {
      speechSynthesis.cancel();
    }
    utteranceRef.current = null;
    activeEngine.current = null;
    setIsSpeaking(false);
    setIsLoading(false);
  }, []);

  // ── Sync state: detect when browser speech ends externally ──
  useEffect(() => {
    if (!isSpeaking) return;

    const interval = setInterval(() => {
      if (activeEngine.current === "browser") {
        if (!speechSynthesis.speaking) {
          stopAll();
        }
      }
      // HTMLAudioElement handles its own `ended` event — no polling needed
    }, 300);

    return () => clearInterval(interval);
  }, [isSpeaking, stopAll]);

  // =========================================================================
  // PATH A — Backend edge-tts (high quality, proper ASEAN language voices)
  // =========================================================================

  const playEdgeTTS = useCallback(
    async (cleanText: string, lang: string): Promise<boolean> => {
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const res = await fetch(`${BACKEND_URL}/api/tts`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: cleanText,
            language: lang,
          }),
          signal: controller.signal,
        });

        if (!res.ok) return false;

        const data = await res.json();
        if (!data.audio_base64) return false;

        // Decode base64 → blob → object URL → play
        const binaryStr = atob(data.audio_base64);
        const bytes = new Uint8Array(binaryStr.length);
        for (let i = 0; i < binaryStr.length; i++) {
          bytes[i] = binaryStr.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: data.content_type || "audio/mpeg" });
        const url = URL.createObjectURL(blob);

        const audio = new Audio(url);
        audioRef.current = audio;
        activeEngine.current = "edge";

        audio.onplay = () => {
          setIsLoading(false);
          setIsSpeaking(true);
        };

        audio.onended = () => {
          URL.revokeObjectURL(url);
          stopAll();
        };

        audio.onerror = () => {
          URL.revokeObjectURL(url);
          stopAll();
        };

        await audio.play();
        return true;
      } catch (err: any) {
        if (err.name === "AbortError") return false;
        return false;
      }
    },
    [stopAll]
  );

  // =========================================================================
  // PATH B — Browser speechSynthesis (offline fallback)
  // =========================================================================

  const playBrowserTTS = useCallback(
    (cleanText: string, lang: string) => {
      if (typeof speechSynthesis === "undefined") return;

      speechSynthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(cleanText);
      utteranceRef.current = utterance;

      const voice = pickVoice(lang);
      if (voice) {
        utterance.voice = voice;
        utterance.lang = voice.lang;
      } else if (lang) {
        utterance.lang = lang;
      }

      utterance.rate = 0.93;
      utterance.pitch = 1.08;

      activeEngine.current = "browser";

      utterance.onstart = () => {
        setIsLoading(false);
        setIsSpeaking(true);
      };

      utterance.onend = () => stopAll();
      utterance.onerror = () => stopAll();

      speechSynthesis.speak(utterance);

      // Chrome hang safety
      setTimeout(() => {
        setIsLoading((prev) => {
          if (prev && !speechSynthesis.speaking) {
            stopAll();
            return false;
          }
          return prev;
        });
      }, 3000);
    },
    [stopAll]
  );

  // =========================================================================
  // Click handler — edge-tts first, browser fallback
  // =========================================================================

  const handleToggle = useCallback(async () => {
    // ── Stop ──
    if (isSpeaking || isLoading) {
      stopAll();
      return;
    }

    // ── Start ──
    const cleanText = stripMarkdown(text);
    if (!cleanText) return;

    // Extract base language for the backend (e.g. "ms-MY" → "ms")
    const lang = langCode || "en";
    const langBase = lang.split("-")[0].toLowerCase();

    setIsLoading(true);

    // Try backend edge-tts first (better quality for ASEAN languages)
    const edgeWorked = await playEdgeTTS(cleanText, langBase);

    if (!edgeWorked && !abortRef.current?.signal.aborted) {
      // Fallback to browser speechSynthesis
      playBrowserTTS(cleanText, lang);
    }
  }, [isSpeaking, isLoading, text, langCode, stopAll, playEdgeTTS, playBrowserTTS]);

  // Don't render while streaming or if no text
  if (isStreaming || !text?.trim()) return null;

  return (
    <button
      onClick={handleToggle}
      className={cn(
        "inline-flex items-center justify-center h-7 w-7 rounded-lg transition-all duration-200",
        "text-slate-400 dark:text-slate-500",
        "hover:text-amber-600 dark:hover:text-amber-400",
        "hover:bg-amber-50 dark:hover:bg-amber-900/30",
        isSpeaking && "text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/30",
        isLoading && "opacity-60 cursor-wait"
      )}
      aria-label={isSpeaking ? "Stop reading aloud" : "Read aloud"}
      title={isSpeaking ? "Stop reading aloud" : "Read aloud"}
    >
      {isLoading ? (
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
      ) : isSpeaking ? (
        <VolumeX className="h-3.5 w-3.5" />
      ) : (
        <Volume2 className="h-3.5 w-3.5" />
      )}
    </button>
  );
}