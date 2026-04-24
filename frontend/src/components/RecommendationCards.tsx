"use client";

import { ExternalLink, MapPin, Phone, Share2, ChevronRight } from "lucide-react";
import Image from "next/image";
import ReactMarkdown from "react-markdown";
import type { Recommendation, RecommendationsPayload } from "@/lib/types";

function handleAction(action: Recommendation["action"]) {
  if (!action || action.type === "none") return;
  switch (action.type) {
    case "navigate":
      if (action.lat != null && action.lng != null) {
        window.open(
          `https://www.google.com/maps/dir/?api=1&destination=${action.lat},${action.lng}`,
          "_blank"
        );
      }
      break;
    case "call":
      if (action.phone) window.open(`tel:${action.phone}`, "_self");
      break;
    case "link":
      if (action.url) window.open(action.url, "_blank");
      break;
    case "share":
      if (navigator.share && action.url) {
        navigator.share({ title: action.label, url: action.url }).catch(() => {});
      }
      break;
  }
}

const ACTION_ICON: Record<string, React.ElementType> = {
  navigate: MapPin,
  call: Phone,
  share: Share2,
  link: ExternalLink,
};

const COUNTRY_FLAG: Record<string, string> = {
  MY: "🇲🇾",
  ID: "🇮🇩",
  PH: "🇵🇭",
  TH: "🇹🇭",
  ASEAN: "🌏",
};

function RecCard({ item }: { item: Recommendation }) {
  const hasAction = item.action && item.action.type !== "none";
  const Icon = hasAction ? ACTION_ICON[item.action!.type] || ChevronRight : ChevronRight;

  return (
    <button
      onClick={() => hasAction && handleAction(item.action)}
      className="w-full text-left bg-white dark:bg-slate-800/90 rounded-xl border border-slate-200/80 dark:border-slate-700/60 p-4 shadow-sm hover:shadow-md hover:border-amber-300 dark:hover:border-amber-600 transition-all active:scale-[0.98] group"
    >
      <div className="flex items-start gap-3">
        {item.icon && (
          <span className="text-2xl leading-none mt-0.5 shrink-0" role="img" aria-hidden>
            {item.icon}
          </span>
        )}

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h4 className="text-sm font-bold text-slate-900 dark:text-white leading-snug truncate">
              {item.title}
            </h4>
            {item.country && COUNTRY_FLAG[item.country] && (
              <span className="text-sm shrink-0">{COUNTRY_FLAG[item.country]}</span>
            )}
          </div>

          <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed line-clamp-2 mb-2">
            {item.description}
          </p>

          {item.eligibility && (
            <p className="text-xs text-emerald-600 dark:text-emerald-400 font-medium">
              {item.eligibility}
            </p>
          )}

          {item.tags && item.tags.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {item.tags.map((tag) => (
                <span
                  key={tag}
                  className="inline-block px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide rounded-md bg-slate-100 dark:bg-slate-700 text-slate-500 dark:text-slate-400"
                >
                  {tag}
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="shrink-0 mt-1 text-slate-400 dark:text-slate-500 group-hover:text-amber-500 dark:group-hover:text-amber-400 transition-colors">
          <Icon className="h-4 w-4" />
        </div>
      </div>
    </button>
  );
}

interface RecommendationCardsProps {
  data: RecommendationsPayload;
  introText?: string;
  compact?: boolean;
}

export default function RecommendationCards({ data, introText, compact = false }: RecommendationCardsProps) {
  const { title, items } = data;

  if (!items || items.length === 0) return null;

  const displayIntro = introText?.trim() || title;

  const cardsContent = (
    <div className="space-y-2">
      {items.map((item, i) => (
        <RecCard key={item.id || `rec-${i}`} item={item} />
      ))}
    </div>
  );

  // Compact: cards only (avatar + intro handled by MessageRouter)
  if (compact) {
    return cardsContent;
  }

  // Full: avatar + intro + cards (for standalone use / test page)
  return (
    <div className="flex w-full min-w-0 mb-8 justify-start">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-amber-100 dark:bg-amber-900/50 mr-4 mt-0.5 border border-amber-200/50 dark:border-amber-800/50 shadow-sm transition-colors duration-500 overflow-hidden">
        <Image src="/icons/cat.png" alt="Ara" width={20} height={20} className="object-contain" />
      </div>

      <div className="max-w-[85%] md:max-w-[75%] w-full min-w-0 overflow-hidden">
        {displayIntro && (
          <div className="text-[16px] leading-relaxed text-slate-800 dark:text-slate-200 mb-2 font-medium ara-markdown break-words overflow-wrap-anywhere">
            <ReactMarkdown
              components={{
                p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                strong: ({ children }) => (
                  <strong className="font-bold text-slate-900 dark:text-white">{children}</strong>
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