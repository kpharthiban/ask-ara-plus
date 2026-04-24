"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { Mic, MicOff, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

// ── Config ───────────────────────────────────────────────────────────────────

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

const SILENCE_THRESHOLD = 15;     // RMS level below this = silence (0–128 scale)
const SILENCE_DURATION_MS = 2000; // auto-stop after 2s of silence
const MAX_RECORDING_MS = 30000;   // hard cap: 30 seconds

// ── Web Speech API types ─────────────────────────────────────────────────────

interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}
interface SpeechRecognitionErrorEvent extends Event {
  error: string;
  message?: string;
}
interface SpeechRecognitionInstance extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  start: () => void;
  stop: () => void;
  abort: () => void;
  onresult: ((e: SpeechRecognitionEvent) => void) | null;
  onerror: ((e: SpeechRecognitionErrorEvent) => void) | null;
  onend: (() => void) | null;
  onstart: (() => void) | null;
}

// ── Props ────────────────────────────────────────────────────────────────────

interface VoiceInputProps {
  langCode: string;
  onTranscript: (text: string) => void;
  onError: (message: string) => void;
  disabled?: boolean;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function getWebSpeechAPI(): (new () => SpeechRecognitionInstance) | null {
  if (typeof window === "undefined") return null;
  return (
    (window as any).SpeechRecognition ||
    (window as any).webkitSpeechRecognition ||
    null
  );
}

// ── Component ────────────────────────────────────────────────────────────────

export default function VoiceInput({ langCode, onTranscript, onError, disabled }: VoiceInputProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isSupported, setIsSupported] = useState(true);

  // Track whether Web Speech API works in this browser.
  // Starts true, flips to false permanently on first "network" / unsupported error.
  const webSpeechWorks = useRef(true);

  // Refs for MediaRecorder path
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const maxTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const analyserLoopRef = useRef<number | null>(null);

  // Ref for Web Speech API path
  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null);

  // Check support after mount
  useEffect(() => {
    const hasMediaRecorder =
      typeof navigator !== "undefined" &&
      !!navigator.mediaDevices?.getUserMedia &&
      typeof MediaRecorder !== "undefined";
    const hasWebSpeech = getWebSpeechAPI() !== null;

    setIsSupported(hasMediaRecorder || hasWebSpeech);

    if (!hasWebSpeech) {
      webSpeechWorks.current = false;
    }
  }, []);

  // ── Cleanup on unmount ──
  useEffect(() => {
    return () => {
      recognitionRef.current?.abort();
      mediaRecorderRef.current?.stop();
      streamRef.current?.getTracks().forEach((t) => t.stop());
      audioContextRef.current?.close();
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
      if (maxTimerRef.current) clearTimeout(maxTimerRef.current);
      if (analyserLoopRef.current) cancelAnimationFrame(analyserLoopRef.current);
    };
  }, []);

  // =========================================================================
  // PATH A — Web Speech API (fast, high quality, Chrome/Edge)
  // =========================================================================

  const startWebSpeech = useCallback(() => {
    const SpeechRecognition = getWebSpeechAPI();
    if (!SpeechRecognition) {
      webSpeechWorks.current = false;
      return false; // signal: fall through to MediaRecorder
    }

    if (recognitionRef.current) recognitionRef.current.abort();

    const recognition = new SpeechRecognition();
    recognitionRef.current = recognition;

    if (langCode) recognition.lang = langCode;
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => setIsRecording(true);

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      const transcript = event.results[0]?.[0]?.transcript;
      if (transcript?.trim()) {
        onTranscript(transcript.trim());
      } else {
        onError("No speech detected — try again");
      }
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      setIsRecording(false);
      recognitionRef.current = null;

      // "network" or "service-not-allowed" means this browser blocks the API
      // (e.g. Brave). Mark it dead and fall back to MediaRecorder.
      if (event.error === "network" || event.error === "service-not-allowed") {
        webSpeechWorks.current = false;
        // Auto-retry with MediaRecorder path
        startMediaRecorder();
        return;
      }

      switch (event.error) {
        case "not-allowed":
          onError("Microphone permission denied");
          break;
        case "no-speech":
          onError("No speech detected — try again");
          break;
        case "aborted":
          break;
        default:
          onError("Voice input error — try again");
      }
    };

    recognition.onend = () => {
      setIsRecording(false);
      recognitionRef.current = null;
    };

    try {
      recognition.start();
      return true; // success
    } catch {
      webSpeechWorks.current = false;
      return false;
    }
  }, [langCode, onTranscript, onError]);

  const stopWebSpeech = useCallback(() => {
    recognitionRef.current?.stop();
    setIsRecording(false);
  }, []);

  // =========================================================================
  // PATH B — MediaRecorder + Groq Whisper (universal fallback)
  // =========================================================================

  // ── Send audio to backend ──
  const transcribeAudio = useCallback(
    async (audioBlob: Blob) => {
      setIsTranscribing(true);
      try {
        const formData = new FormData();
        formData.append("audio", audioBlob, "recording.webm");
        if (langCode) {
          formData.append("language", langCode.split("-")[0]);
        }

        const res = await fetch(`${BACKEND_URL}/api/transcribe`, {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => null);
          throw new Error(errData?.detail || `Transcription failed (${res.status})`);
        }

        const data = await res.json();
        const transcript = data.text?.trim();

        if (transcript) {
          onTranscript(transcript);
        } else {
          onError("No speech detected — try again");
        }
      } catch (err: any) {
        onError(err.message || "Transcription failed — try again");
      } finally {
        setIsTranscribing(false);
      }
    },
    [langCode, onTranscript, onError]
  );

  // ── Silence detection via AnalyserNode ──
  const startSilenceDetection = useCallback(
    (stream: MediaStream) => {
      try {
        const audioContext = new AudioContext();
        audioContextRef.current = audioContext;
        const source = audioContext.createMediaStreamSource(stream);
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 512;
        source.connect(analyser);

        const dataArray = new Uint8Array(analyser.fftSize);
        let silenceStart: number | null = null;

        const checkSilence = () => {
          // Bail if no longer recording
          if (!mediaRecorderRef.current || mediaRecorderRef.current.state !== "recording") return;

          analyser.getByteTimeDomainData(dataArray);

          // Calculate RMS
          let sum = 0;
          for (let i = 0; i < dataArray.length; i++) {
            const val = (dataArray[i] - 128) / 128;
            sum += val * val;
          }
          const rms = Math.sqrt(sum / dataArray.length) * 128;

          if (rms < SILENCE_THRESHOLD) {
            if (silenceStart === null) {
              silenceStart = Date.now();
            } else if (Date.now() - silenceStart > SILENCE_DURATION_MS) {
              // Silence exceeded threshold — auto-stop
              stopMediaRecorder();
              return;
            }
          } else {
            silenceStart = null; // reset on any sound
          }

          analyserLoopRef.current = requestAnimationFrame(checkSilence);
        };

        analyserLoopRef.current = requestAnimationFrame(checkSilence);
      } catch {
        // Silence detection is a nice-to-have; don't fail recording if it errors
      }
    },
    [] // stopMediaRecorder added via ref call below
  );

  // ── Stop MediaRecorder ──
  const stopMediaRecorder = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);

    // Cleanup silence detection
    if (analyserLoopRef.current) cancelAnimationFrame(analyserLoopRef.current);
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    if (maxTimerRef.current) clearTimeout(maxTimerRef.current);
    audioContextRef.current?.close().catch(() => {});
    audioContextRef.current = null;
  }, []);

  // ── Start MediaRecorder ──
  const startMediaRecorder = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
          ? "audio/webm"
          : "";

      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        streamRef.current = null;

        const audioBlob = new Blob(chunksRef.current, {
          type: mimeType || "audio/webm",
        });
        chunksRef.current = [];

        if (audioBlob.size > 1000) {
          transcribeAudio(audioBlob);
        } else {
          onError("Recording too short — hold the mic longer");
        }
      };

      recorder.start();
      setIsRecording(true);

      // Start silence detection
      startSilenceDetection(stream);

      // Hard cap safety timer
      maxTimerRef.current = setTimeout(() => {
        stopMediaRecorder();
      }, MAX_RECORDING_MS);

    } catch (err: any) {
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        onError("Microphone permission denied");
      } else if (err.name === "NotFoundError") {
        onError("No microphone found");
      } else {
        onError("Could not start recording");
      }
    }
  }, [transcribeAudio, onError, startSilenceDetection, stopMediaRecorder]);

  // =========================================================================
  // Click handler — picks the right path
  // =========================================================================

  const handleClick = useCallback(() => {
    if (isRecording) {
      // Stop whichever method is active
      if (recognitionRef.current) {
        stopWebSpeech();
      } else {
        stopMediaRecorder();
      }
      return;
    }

    if (isTranscribing) return;

    // Try Web Speech API first; if it fails, fall back
    if (webSpeechWorks.current) {
      const started = startWebSpeech();
      if (!started) {
        // Immediate fallback
        startMediaRecorder();
      }
    } else {
      startMediaRecorder();
    }
  }, [isRecording, isTranscribing, startWebSpeech, stopWebSpeech, startMediaRecorder, stopMediaRecorder]);

  // =========================================================================
  // Render
  // =========================================================================

  return (
    <Button
      type="button"
      size="icon"
      disabled={disabled || !isSupported || isTranscribing}
      onClick={handleClick}
      className={cn(
        "relative rounded-xl h-12 w-12 shrink-0 transition-all duration-200",
        isRecording
          ? "bg-red-500 dark:bg-red-500 hover:bg-red-600 dark:hover:bg-red-400 text-white"
          : isTranscribing
            ? "bg-amber-100 dark:bg-amber-900/40 text-amber-600 dark:text-amber-400"
            : "bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300",
        !isSupported && "opacity-40 cursor-not-allowed"
      )}
      aria-label={
        isRecording ? "Stop recording" : isTranscribing ? "Transcribing..." : "Start voice input"
      }
    >
      {/* Pulsing ring while recording */}
      {isRecording && (
        <span className="absolute inset-0 rounded-xl border-2 border-red-400 animate-ping opacity-50" />
      )}

      {isTranscribing ? (
        <Loader2 className="h-5 w-5 animate-spin" />
      ) : isRecording ? (
        <MicOff className="h-5 w-5 relative z-10" />
      ) : (
        <Mic className="h-5 w-5" />
      )}
    </Button>
  );
}