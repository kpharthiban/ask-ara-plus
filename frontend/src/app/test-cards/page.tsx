"use client";

/**
 * /app/test-cards/page.tsx
 *
 * Visual test page for StepCards + RecommendationCards.
 * Visit http://localhost:3000/test-cards — no backend needed.
 * DELETE THIS FILE before submission.
 */

import { useState } from "react";
import StepCards from "@/components/StepCards";
import RecommendationCards from "@/components/RecommendationCards";
import MessageBubble from "@/components/MessageBubble";
import type {
  Message,
  StepCardsPayload,
  RecommendationsPayload,
} from "@/lib/types";

// ── Dummy: Step Cards (3-card) ────────────────────────────────────────────

const DEMO_STEP_CARDS: StepCardsPayload = {
  type: "step_cards",
  summary: "Berikut adalah langkah-langkah untuk memohon Bantuan Wang Ihsan:",
  cards: [
    {
      step: 1,
      total: 3,
      title: "Pergi ke Pejabat JKM",
      icon: "📋",
      body: "Lawati pejabat JKM yang berdekatan dengan rumah anda untuk mendapatkan borang permohonan.",
      location: "Pejabat JKM Kota Bharu",
      hours: "Isnin–Jumaat, 8:30pg – 4:30ptg",
      action: {
        type: "navigate",
        label: "Navigasi ke JKM",
        lat: 6.12,
        lng: 102.24,
      },
    },
    {
      step: 2,
      total: 3,
      title: "Bawa dokumen ini",
      icon: "📄",
      body: "Pastikan anda membawa semua dokumen yang diperlukan.",
      checklist: [
        "IC (MyKad) — asal dan salinan",
        "Bukti alamat (bil utiliti)",
        "Penyata bank 3 bulan terakhir",
        "Gambar kerosakan rumah (jika ada)",
      ],
      action: {
        type: "link",
        label: "Tiada IC? Ketahui lagi",
        url: "https://www.jpn.gov.my",
      },
    },
    {
      step: 3,
      total: 3,
      title: "Hantar permohonan",
      icon: "✅",
      body: "Serahkan borang yang lengkap kepada pegawai JKM. Keputusan dalam 14 hari bekerja.",
      deadline: "Tarikh akhir: 30 Mac 2026 (15 hari lagi)",
      amount: "Anda boleh dapat: RM500",
      action: {
        type: "call",
        label: "Hubungi JKM: 1800-22-8000",
        phone: "1800228000",
      },
    },
  ],
};

// ── Dummy: Recommendations ───────────────────────────────────────────────

const DEMO_RECOMMENDATIONS: RecommendationsPayload = {
  type: "recommendations",
  title: "Berdasarkan profil anda, anda mungkin layak untuk:",
  items: [
    {
      id: "bwi",
      title: "Bantuan Wang Ihsan (BWI)",
      description: "Bantuan kewangan RM500 untuk mangsa banjir bagi membaiki kerosakan rumah.",
      icon: "🌊",
      country: "MY",
      eligibility: "Layak — pendapatan isi rumah < RM5,000/bulan",
      tags: ["flood_relief", "jkm"],
      action: { type: "link", label: "Mohon sekarang", url: "#" },
    },
    {
      id: "skim-perkeso",
      title: "Skim Bencana Pekerjaan (SOCSO)",
      description: "Perlindungan insurans untuk pekerja yang cedera di tempat kerja.",
      icon: "👷",
      country: "MY",
      eligibility: "Layak — pekerja berdaftar SOCSO",
      tags: ["worker_rights", "socso"],
      action: { type: "link", label: "Semak kelayakan", url: "#" },
    },
    {
      id: "bpjs-kesehatan",
      title: "BPJS Kesehatan",
      description: "Program jaminan kesehatan nasional Indonesia.",
      icon: "🏥",
      country: "ID",
      eligibility: "Layak — warganegara Indonesia",
      tags: ["health", "bpjs"],
      action: { type: "link", label: "Daftar online", url: "#" },
    },
  ],
};

// ── Dummy messages ────────────────────────────────────────────────────────

const DEMO_USER_MESSAGE: Message = {
  id: "demo-user",
  sender: "user",
  content: "Macam mana nak mohon bantuan banjir JKM?",
  contentType: "text",
  timestamp: new Date(),
};

const DEMO_TEXT_MESSAGE: Message = {
  id: "demo-text",
  sender: "ara",
  content:
    "SOCSO (juga dikenali sebagai **PERKESO**) ialah organisasi keselamatan sosial Malaysia.\n\nAnda boleh membuat tuntutan melalui portal rasmi atau di pejabat SOCSO berdekatan.",
  contentType: "text",
  timestamp: new Date(),
  toolCalls: ["search_documents", "simplify"],
  sources: [
    { title: "SOCSO — Panduan Pekerja", country: "MY", url: "#" },
    { title: "Employment Act 1955 Summary", country: "MY" },
  ],
};

// ── Test page ─────────────────────────────────────────────────────────────

export default function TestCardsPage() {
  const [activeDemo, setActiveDemo] = useState<
    "all" | "steps" | "recs" | "text"
  >("all");

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 transition-colors">
      {/* Header */}
      <div className="sticky top-0 z-50 bg-white/90 dark:bg-slate-950/90 backdrop-blur-md border-b border-slate-200 dark:border-slate-800 px-4 py-3">
        <h1 className="text-lg font-bold text-slate-900 dark:text-white text-center">
          AskAra<span className="text-teal-600 dark:text-teal-400">+</span>{" "}
          <span className="text-sm font-normal text-slate-500">
            Component Test
          </span>
        </h1>
        <div className="flex justify-center gap-2 mt-2 flex-wrap">
          {(
            [
              ["all", "All"],
              ["steps", "Step Cards"],
              ["recs", "Recommendations"],
              ["text", "Text Bubble"],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              onClick={() => setActiveDemo(key)}
              className={`px-3 py-1 text-xs font-semibold rounded-lg transition-colors ${
                activeDemo === key
                  ? "bg-teal-600 text-white"
                  : "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Content — mimics ChatWindow layout */}
      <div className="max-w-3xl mx-auto px-4 md:px-8 pt-8 pb-32 bg-dot-pattern">
        {(activeDemo === "all" || activeDemo === "steps") && (
          <>
            <MessageBubble message={DEMO_USER_MESSAGE} />
            <StepCards
              data={DEMO_STEP_CARDS}
              introText="Berikut adalah langkah-langkah untuk memohon Bantuan Wang Ihsan daripada JKM:"
            />
          </>
        )}

        {(activeDemo === "all" || activeDemo === "recs") && (
          <RecommendationCards
            data={DEMO_RECOMMENDATIONS}
            introText="Berdasarkan profil anda, anda mungkin layak untuk program-program berikut:"
          />
        )}

        {(activeDemo === "all" || activeDemo === "text") && (
          <MessageBubble message={DEMO_TEXT_MESSAGE} />
        )}
      </div>
    </div>
  );
}