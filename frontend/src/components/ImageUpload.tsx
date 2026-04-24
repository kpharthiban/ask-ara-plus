"use client";

import { useRef, useCallback } from "react";
import { ImagePlus } from "lucide-react";
import { Button } from "@/components/ui/button";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface SelectedImage {
  /** Raw base64 string — NO data-URI prefix. */
  base64: string;
  /** Full data-URI for <img> src (e.g. "data:image/jpeg;base64,..."). */
  previewUrl: string;
  /** MIME type detected from the file (image/jpeg, image/png, …). */
  mediaType: string;
}

interface ImageUploadProps {
  onImageSelected: (image: SelectedImage) => void;
  onError: (message: string) => void;
  disabled?: boolean;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function ImageUpload({
  onImageSelected,
  onError,
  disabled,
}: ImageUploadProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);

  const handleClick = useCallback(() => {
    inputRef.current?.click();
  }, []);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      // Reset so the same file can be re-selected if needed
      e.target.value = "";

      // Guard: images only
      if (!file.type.startsWith("image/")) {
        onError("Please select an image file.");
        return;
      }

      // Guard: 10 MB max
      if (file.size > 10 * 1024 * 1024) {
        onError("Image is too large. Please choose one under 10 MB.");
        return;
      }

      const reader = new FileReader();

      reader.onload = () => {
        const dataUrl = reader.result as string;
        // "data:<mediaType>;base64,<b64>" → split on first comma
        const [prefix, base64] = dataUrl.split(",");
        const mediaType = prefix.replace("data:", "").replace(";base64", "");

        onImageSelected({
          base64,
          previewUrl: dataUrl,
          mediaType,
        });
      };

      reader.onerror = () => {
        onError("Could not read the image file. Please try again.");
      };

      reader.readAsDataURL(file);
    },
    [onImageSelected, onError]
  );

  return (
    <>
      {/* Hidden file input */}
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleFileChange}
        aria-hidden="true"
        tabIndex={-1}
      />

      {/* Visible button — sits inside the input bar */}
      <Button
        type="button"
        variant="ghost"
        size="icon"
        disabled={disabled}
        onClick={handleClick}
        className="rounded-xl h-12 w-12 shrink-0 text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
        aria-label="Upload image"
        title="Upload image"
      >
        <ImagePlus className="h-5 w-5" />
      </Button>
    </>
  );
}