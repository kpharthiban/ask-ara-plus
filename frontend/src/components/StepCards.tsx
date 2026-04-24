"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  MapPin,
  Phone,
  Share2,
  ExternalLink,
  ChevronLeft,
  ChevronRight,
  CheckSquare,
  Clock,
  CalendarClock,
  Banknote,
} from "lucide-react";
import Image from "next/image";
import ReactMarkdown from "react-markdown";
import type { StepCard, StepCardsPayload } from "@/lib/types";

// ── Action button handler ─────────────────────────────────────────────────

function handleAction(action: StepCard["action"]) {
  if (!action || action.type === "none") return;

  switch (action.type) {
    case "navigate":
      if (action.lat != null && action.lng != null) {
        window.open(
          `https://www.google.com/maps/dir/?api=1&destination=${action.lat},${action.lng}`,
          "_blank"
        );
      } else if (action.url) {
        window.open(action.url, "_blank");
      }
      break;
    case "call":
      if (action.phone) window.open(`tel:${action.phone}`, "_self");
      break;
    case "share":
      if (navigator.share && action.url) {
        navigator.share({ title: action.label, url: action.url }).catch(() => {});
      }
      break;
    case "link":
      if (action.url) window.open(action.url, "_blank");
      break;
  }
}

// ── Action button config ──────────────────────────────────────────────────

const ACTION_CONFIG: Record<
  string,
  { icon: React.ElementType; className: string }
> = {
  navigate: { icon: MapPin,       className: "bg-blue-600 hover:bg-blue-700 text-white" },
  call:     { icon: Phone,        className: "bg-emerald-600 hover:bg-emerald-700 text-white" },
  share:    { icon: Share2,       className: "bg-violet-600 hover:bg-violet-700 text-white" },
  link:     { icon: ExternalLink, className: "bg-amber-600 hover:bg-amber-700 text-white" },
};

// ── Individual card ───────────────────────────────────────────────────────
// widthPx is measured by ResizeObserver on the scroll container — explicit
// pixels bypass percentage-resolution quirks in scrollable flex rows where
// `w-full` can resolve against scrollable content width instead of visible width.

function Card({ card, widthPx }: { card: StepCard; widthPx: number }) {
  const actionCfg = card.action && card.action.type !== "none"
    ? ACTION_CONFIG[card.action.type]
    : null;
  const ActionIcon = actionCfg?.icon ?? ExternalLink;

  return (
    <div
      className="step-card-item flex-none snap-center px-1 box-border"
      style={{ width: widthPx > 0 ? `${widthPx}px` : "100%" }}
    >
      <div className="relative bg-white dark:bg-slate-800/90 rounded-2xl border border-slate-200/80 dark:border-slate-700/60 shadow-md overflow-hidden transition-colors duration-300 w-full">

        {/* Top accent bar */}
        <div className="h-1 bg-gradient-to-r from-amber-500 via-amber-400 to-yellow-400" />

        <div className="p-5">
          {/* Step badge + icon */}
          <div className="flex items-center justify-between mb-3">
            <span className="inline-flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-amber-700 dark:text-amber-400 shrink-0">
              <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-amber-100 dark:bg-amber-900/60 text-sm font-extrabold">
                {card.step}
              </span>
              <span className="opacity-60">/ {card.total}</span>
            </span>
            {card.icon && (
              <span className="text-2xl leading-none shrink-0" role="img" aria-hidden>
                {card.icon}
              </span>
            )}
          </div>

          {/* Title */}
          <h3 className="text-lg font-bold text-slate-900 dark:text-white leading-snug mb-2 break-words">
            {card.title}
          </h3>

          {/* Body text */}
          {card.body && (
            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed mb-3 break-words [overflow-wrap:anywhere]">
              {card.body}
            </p>
          )}

          {/* Location — break-all handles long URLs */}
          {card.location && (
            <div className="flex items-start gap-2 text-sm text-slate-600 dark:text-slate-400 mb-2 min-w-0">
              <MapPin className="h-4 w-4 mt-0.5 shrink-0 text-amber-500" />
              <span className="break-all min-w-0 [overflow-wrap:anywhere]">{card.location}</span>
            </div>
          )}

          {/* Hours */}
          {card.hours && (
            <div className="flex items-start gap-2 text-sm text-slate-600 dark:text-slate-400 mb-2 min-w-0">
              <Clock className="h-4 w-4 mt-0.5 shrink-0 text-amber-500" />
              <span className="break-words min-w-0">{card.hours}</span>
            </div>
          )}

          {/* Deadline */}
          {card.deadline && (
            <div className="flex items-start gap-2 text-sm text-slate-600 dark:text-slate-400 mb-2 min-w-0">
              <CalendarClock className="h-4 w-4 mt-0.5 shrink-0 text-amber-500" />
              <span className="font-medium break-words min-w-0">{card.deadline}</span>
            </div>
          )}

          {/* Amount */}
          {card.amount && (
            <div className="flex items-start gap-2 text-sm mb-2 min-w-0">
              <Banknote className="h-4 w-4 mt-0.5 shrink-0 text-emerald-500" />
              <span className="font-bold text-emerald-700 dark:text-emerald-400 break-words min-w-0">
                {card.amount}
              </span>
            </div>
          )}

          {/* Checklist */}
          {card.checklist && card.checklist.length > 0 && (
            <div className="mt-3 space-y-1.5">
              {card.checklist.map((item, i) => (
                <div
                  key={i}
                  className="flex items-start gap-2 text-sm text-slate-700 dark:text-slate-300 min-w-0"
                >
                  <CheckSquare className="h-4 w-4 mt-0.5 shrink-0 text-amber-500" />
                  <span className="break-words min-w-0 flex-1 [overflow-wrap:anywhere]">{item}</span>
                </div>
              ))}
            </div>
          )}

          {/* Action button */}
          {actionCfg && card.action && (
            <button
              onClick={() => handleAction(card.action)}
              className={`mt-4 w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl text-sm font-bold transition-all active:scale-[0.97] ${actionCfg.className}`}
            >
              <ActionIcon className="h-4 w-4 shrink-0" />
              <span className="truncate">{card.action.label || card.action.type}</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Main StepCards component ──────────────────────────────────────────────

interface StepCardsProps {
  data: StepCardsPayload;
  introText?: string;
  compact?: boolean;
}

export default function StepCards({ data, introText, compact = false }: StepCardsProps) {
  const { cards, summary } = data;
  const scrollRef = useRef<HTMLDivElement>(null);
  const [activeIndex, setActiveIndex] = useState(0);

  // ── Measured card width ───────────────────────────────────────────────
  // ResizeObserver watches the scroll container and records its clientWidth
  // (the visible pixel width, NOT the scrollable content width). This value
  // is passed directly to each Card as an explicit style, making card sizing
  // immune to any CSS percentage-resolution bugs in scrollable flex rows.
  const [cardWidthPx, setCardWidthPx] = useState(0);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const measure = () => setCardWidthPx(el.clientWidth);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const displayIntro = introText?.trim() || summary;

  // Use measured width for accurate index calculation
  const updateActiveIndex = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const w = cardWidthPx || el.clientWidth;
    if (w === 0) return;
    const idx = Math.round(el.scrollLeft / w);
    setActiveIndex(Math.max(0, Math.min(idx, cards.length - 1)));
  }, [cards.length, cardWidthPx]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.addEventListener("scroll", updateActiveIndex, { passive: true });
    return () => el.removeEventListener("scroll", updateActiveIndex);
  }, [updateActiveIndex]);

  const scrollTo = (index: number) => {
    const el = scrollRef.current;
    if (!el) return;
    const w = cardWidthPx || el.clientWidth;
    const clamped = Math.max(0, Math.min(index, cards.length - 1));
    el.scrollTo({ left: clamped * w, behavior: "smooth" });
  };

  if (!cards || cards.length === 0) return null;

  // ── Shared carousel UI ────────────────────────────────────────────────
  const cardsContent = (
    <>
      {/* Outer div: positioning context for nav arrows only.
          NO overflow-hidden — the inner scrollRef's overflow-x:auto already
          clips the carousel. Adding overflow-hidden here would clip the arrow
          buttons that sit at left-1 / right-1. */}
      <div className="relative w-full">
        <div
          ref={scrollRef}
          className="flex overflow-x-auto snap-x snap-mandatory scrollbar-hide"
          style={{
            scrollbarWidth: "none",
            msOverflowStyle: "none",
            WebkitOverflowScrolling: "touch",
            maxWidth: "100%",        // never expand parent
            contain: "layout",       // CSS containment — strict layout isolation
          }}
        >
          {cards.map((card, i) => (
            <Card key={`step-${card.step}-${i}`} card={card} widthPx={cardWidthPx} />
          ))}
        </div>

        {/* Desktop prev/next arrows */}
        {cards.length > 1 && (
          <>
            {activeIndex > 0 && (
              <button
                onClick={() => scrollTo(activeIndex - 1)}
                className="hidden md:flex absolute left-1 top-1/2 -translate-y-1/2 h-8 w-8 items-center justify-center rounded-full bg-white/90 dark:bg-slate-700/90 border border-slate-200 dark:border-slate-600 shadow-md text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-600 transition-all z-10 backdrop-blur-sm"
                aria-label="Previous step"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
            )}
            {activeIndex < cards.length - 1 && (
              <button
                onClick={() => scrollTo(activeIndex + 1)}
                className="hidden md:flex absolute right-1 top-1/2 -translate-y-1/2 h-8 w-8 items-center justify-center rounded-full bg-white/90 dark:bg-slate-700/90 border border-slate-200 dark:border-slate-600 shadow-md text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-600 transition-all z-10 backdrop-blur-sm"
                aria-label="Next step"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            )}
          </>
        )}
      </div>

      {/* Progress dots */}
      {cards.length > 1 && (
        <div className="flex items-center justify-center gap-1.5 mt-3">
          {cards.map((_, i) => (
            <button
              key={i}
              onClick={() => scrollTo(i)}
              aria-label={`Go to step ${i + 1}`}
              className={`rounded-full transition-all duration-300 ${
                i === activeIndex
                  ? "w-6 h-2 bg-amber-500 dark:bg-amber-400"
                  : "w-2 h-2 bg-slate-300 dark:bg-slate-600 hover:bg-slate-400 dark:hover:bg-slate-500"
              }`}
            />
          ))}
        </div>
      )}

      {/* Mobile swipe hint */}
      {cards.length > 1 && activeIndex === 0 && (
        <p className="text-center text-[11px] text-slate-400 dark:text-slate-500 mt-2 md:hidden animate-pulse">
          Swipe to see all {cards.length} steps →
        </p>
      )}
    </>
  );

  // ── Compact mode (used by MessageRouter in ChatWindow) ────────────────
  if (compact) {
    return cardsContent;
  }

  // ── Full mode (standalone / test page) ───────────────────────────────
  return (
    <div className="flex w-full min-w-0 mb-8 justify-start">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-amber-100 dark:bg-amber-900/50 mr-4 mt-0.5 border border-amber-200/50 dark:border-amber-800/50 shadow-sm transition-colors duration-500 overflow-hidden">
        <Image src="/icons/cat.png" alt="Ara" width={20} height={20} className="object-contain" />
      </div>

      <div className="max-w-[85%] md:max-w-[75%] w-full min-w-0 overflow-hidden">
        {displayIntro && (
          <div className="text-[16px] leading-relaxed text-slate-800 dark:text-slate-200 mb-3 font-medium ara-markdown break-words overflow-wrap-anywhere">
            <ReactMarkdown
              components={{
                p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                strong: ({ children }) => (
                  <strong className="font-bold text-slate-900 dark:text-white">{children}</strong>
                ),
                a: ({ href, children }) => (
                  <a href={href} target="_blank" rel="noopener noreferrer"
                    className="text-amber-600 dark:text-amber-400 underline underline-offset-2">
                    {children}
                  </a>
                ),
              }}
            >
              {displayIntro}
            </ReactMarkdown>
          </div>
        )}
        {cardsContent}
      </div>
    </div>
  );
}