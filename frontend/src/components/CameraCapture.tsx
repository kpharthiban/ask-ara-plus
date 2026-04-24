"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { Camera, X, RotateCcw, Send, Loader2, SwitchCamera } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

// ── Config ───────────────────────────────────────────────────────────────────

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

// ── Props ────────────────────────────────────────────────────────────────────

interface CameraCaptureProps {
  /** Called with the extracted text + original base64 thumbnail after scan completes */
  onCapture: (result: {
    extractedText: string;
    thumbnailBase64: string;
    documentType: string;
    issuingAgency: string;
  }) => void;
  onError: (message: string) => void;
  disabled?: boolean;
}

// ── Component ────────────────────────────────────────────────────────────────

export default function CameraCapture({ onCapture, onError, disabled }: CameraCaptureProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isSupported, setIsSupported] = useState(true);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [facingMode, setFacingMode] = useState<"environment" | "user">("environment");

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Check support
  useEffect(() => {
    setIsSupported(
      typeof navigator !== "undefined" &&
      !!navigator.mediaDevices?.getUserMedia
    );
  }, []);

  // ── Start camera stream ──
  const startCamera = useCallback(async (facing: "environment" | "user" = facingMode) => {
    try {
      // Stop existing stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: facing,
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch (err: any) {
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        onError("Camera permission denied");
      } else if (err.name === "NotFoundError") {
        onError("No camera found");
      } else {
        onError("Could not start camera");
      }
      setIsOpen(false);
    }
  }, [facingMode, onError]);

  // ── Stop camera stream ──
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  // ── Open overlay ──
  const handleOpen = useCallback(() => {
    setCapturedImage(null);
    setIsOpen(true);
    // Small delay to let the overlay render before starting camera
    setTimeout(() => startCamera(), 100);
  }, [startCamera]);

  // ── Close overlay ──
  const handleClose = useCallback(() => {
    stopCamera();
    setCapturedImage(null);
    setIsScanning(false);
    setIsOpen(false);
  }, [stopCamera]);

  // ── Capture frame ──
  const handleCapture = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0);

    // Convert to JPEG base64 (good compression for documents)
    const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
    setCapturedImage(dataUrl);

    // Stop camera while previewing
    stopCamera();
  }, [stopCamera]);

  // ── Retake ──
  const handleRetake = useCallback(() => {
    setCapturedImage(null);
    startCamera();
  }, [startCamera]);

  // ── Switch camera ──
  const handleSwitchCamera = useCallback(() => {
    const newFacing = facingMode === "environment" ? "user" : "environment";
    setFacingMode(newFacing);
    startCamera(newFacing);
  }, [facingMode, startCamera]);

  // ── Send to backend for scanning ──
  const handleSend = useCallback(async () => {
    if (!capturedImage) return;

    setIsScanning(true);

    try {
      // Strip the data URI prefix to get raw base64
      const base64 = capturedImage.split(",")[1];

      const res = await fetch(`${BACKEND_URL}/api/scan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: base64 }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => null);
        throw new Error(err?.detail || `Scan failed (${res.status})`);
      }

      const data = await res.json();

      if (!data.extracted_text || data.extracted_text.trim().length === 0) {
        throw new Error("Could not read any text from the image. Try a clearer photo.");
      }

      // Create a small thumbnail for the chat (resize to ~200px wide)
      const thumbnail = await createThumbnail(capturedImage, 200);

      onCapture({
        extractedText: data.extracted_text,
        thumbnailBase64: thumbnail,
        documentType: data.document_type || "unknown",
        issuingAgency: data.issuing_agency || "",
      });

      handleClose();
    } catch (err: any) {
      onError(err.message || "Document scan failed — try again");
      setIsScanning(false);
    }
  }, [capturedImage, onCapture, onError, handleClose]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  // Don't render button if camera not supported
  if (!isSupported) return null;

  return (
    <>
      {/* Camera button — sits in the input bar */}
      <Button
        type="button"
        variant="ghost"
        size="icon"
        disabled={disabled}
        onClick={handleOpen}
        className="rounded-xl h-12 w-12 shrink-0 text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
        aria-label="Scan document"
        title="Scan document"
      >
        <Camera className="h-5 w-5" />
      </Button>

      {/* ── Fullscreen camera overlay ── */}
      {isOpen && (
        <div className="fixed inset-0 z-50 bg-black flex flex-col">
          {/* Top bar */}
          <div className="flex items-center justify-between px-4 py-3 bg-black/80 backdrop-blur-sm">
            <button
              onClick={handleClose}
              className="p-2 rounded-full text-white/80 hover:text-white hover:bg-white/10 transition-colors"
              aria-label="Close camera"
            >
              <X className="h-6 w-6" />
            </button>
            <p className="text-white/80 text-sm font-medium">
              {capturedImage ? "Preview" : "Scan Document"}
            </p>
            {!capturedImage ? (
              <button
                onClick={handleSwitchCamera}
                className="p-2 rounded-full text-white/80 hover:text-white hover:bg-white/10 transition-colors"
                aria-label="Switch camera"
              >
                <SwitchCamera className="h-5 w-5" />
              </button>
            ) : (
              <div className="w-10" /> /* spacer */
            )}
          </div>

          {/* Camera view / Preview */}
          <div className="flex-1 relative flex items-center justify-center overflow-hidden">
            {capturedImage ? (
              /* Preview captured image */
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={capturedImage}
                alt="Captured document"
                className="max-w-full max-h-full object-contain"
              />
            ) : (
              /* Live camera feed */
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="max-w-full max-h-full object-contain"
              />
            )}

            {/* Scanning overlay */}
            {isScanning && (
              <div className="absolute inset-0 bg-black/50 flex flex-col items-center justify-center gap-3">
                <Loader2 className="h-10 w-10 text-amber-400 animate-spin" />
                <p className="text-white/90 text-sm font-medium">Analyzing document...</p>
              </div>
            )}

            {/* Document guide overlay (when camera is live) */}
            {!capturedImage && !isScanning && (
              <div className="absolute inset-8 border-2 border-white/30 rounded-2xl pointer-events-none">
                <div className="absolute -top-6 left-1/2 -translate-x-1/2 text-white/50 text-xs whitespace-nowrap">
                  Position document within the frame
                </div>
              </div>
            )}
          </div>

          {/* Bottom action bar */}
          <div className="flex items-center justify-center gap-6 px-4 py-6 bg-black/80 backdrop-blur-sm pb-safe">
            {capturedImage ? (
              <>
                {/* Retake */}
                <button
                  onClick={handleRetake}
                  disabled={isScanning}
                  className={cn(
                    "flex flex-col items-center gap-1 text-white/80 hover:text-white transition-colors",
                    isScanning && "opacity-40 cursor-not-allowed"
                  )}
                >
                  <div className="h-14 w-14 rounded-full border-2 border-white/30 flex items-center justify-center">
                    <RotateCcw className="h-6 w-6" />
                  </div>
                  <span className="text-xs">Retake</span>
                </button>

                {/* Send / Scan */}
                <button
                  onClick={handleSend}
                  disabled={isScanning}
                  className={cn(
                    "flex flex-col items-center gap-1 transition-colors",
                    isScanning
                      ? "text-amber-400/60 cursor-wait"
                      : "text-amber-400 hover:text-amber-300"
                  )}
                >
                  <div className="h-16 w-16 rounded-full bg-amber-500 flex items-center justify-center shadow-lg shadow-amber-500/30">
                    {isScanning ? (
                      <Loader2 className="h-7 w-7 text-white animate-spin" />
                    ) : (
                      <Send className="h-7 w-7 text-white" />
                    )}
                  </div>
                  <span className="text-xs">{isScanning ? "Scanning..." : "Scan"}</span>
                </button>
              </>
            ) : (
              /* Capture button */
              <button
                onClick={handleCapture}
                className="flex flex-col items-center gap-1 text-white"
              >
                <div className="h-18 w-18 rounded-full border-4 border-white p-1">
                  <div className="h-full w-full rounded-full bg-white active:bg-white/80 transition-colors" style={{ width: 56, height: 56 }} />
                </div>
                <span className="text-xs text-white/70 mt-1">Capture</span>
              </button>
            )}
          </div>

          {/* Hidden canvas for capture */}
          <canvas ref={canvasRef} className="hidden" />
        </div>
      )}
    </>
  );
}

// ── Helper: create a small thumbnail ─────────────────────────────────────────

async function createThumbnail(dataUrl: string, maxWidth: number): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      const scale = Math.min(maxWidth / img.width, 1);
      canvas.width = img.width * scale;
      canvas.height = img.height * scale;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      }
      resolve(canvas.toDataURL("image/jpeg", 0.6));
    };
    img.src = dataUrl;
  });
}