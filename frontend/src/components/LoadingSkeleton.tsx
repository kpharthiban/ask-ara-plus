"use client";

import Image from "next/image";

/**
 * Skeleton placeholder — shown briefly when the agent is thinking
 * but no tokens have arrived yet. Mimics the shape of a response bubble.
 */
export default function LoadingSkeleton() {
  return (
    <div className="flex w-full min-w-0 mb-8 justify-start animate-in fade-in duration-300">
      {/* Avatar */}
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-amber-100 dark:bg-amber-900/50 mr-4 mt-0.5 border border-amber-200/50 dark:border-amber-800/50 shadow-sm overflow-hidden">
        <Image src="/icons/cat.png" alt="Ara" width={20} height={20} className="object-contain animate-pulse" />
      </div>

      {/* Skeleton lines */}
      <div className="flex-1 max-w-[65%] space-y-2.5 py-1">
        <div className="h-3.5 w-[85%] rounded-full bg-slate-200 dark:bg-slate-800 animate-pulse" />
        <div className="h-3.5 w-[70%] rounded-full bg-slate-200 dark:bg-slate-800 animate-pulse" style={{ animationDelay: "75ms" }} />
        <div className="h-3.5 w-[55%] rounded-full bg-slate-150 dark:bg-slate-800/70 animate-pulse" style={{ animationDelay: "150ms" }} />
      </div>
    </div>
  );
}
