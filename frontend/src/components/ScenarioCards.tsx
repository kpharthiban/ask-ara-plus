"use client";

import { cn } from "@/lib/utils";

// ── Scenario definitions ─────────────────────────────────────────────────────

export interface Scenario {
  id: string;
  icon: string;
  label: string;
  description: string;
  prompt: string;   // message sent to Ara when tapped
}

const SCENARIOS: Scenario[] = [
  {
    id: "worker",
    icon: "👷",
    label: "Worker Rights",
    description: "PERKESO/SOCSO, EPF/KWSP, JTK complaints",
    prompt: "What are my rights as a worker in Malaysia? How do I claim SOCSO/PERKESO benefits or file a complaint with JTK (Jabatan Tenaga Kerja)?",
  },
  {
    id: "medical",
    icon: "🏥",
    label: "Medical & FOMEMA",
    description: "FOMEMA screening, MySejahtera, clinic aid",
    prompt: "I need help with medical services in Malaysia — FOMEMA health screening for foreign workers, MySejahtera, or medical financial aid programs.",
  },
  {
    id: "flood",
    icon: "🌊",
    label: "Flood & Disaster Aid",
    description: "JKM bantuan bencana, evacuation relief",
    prompt: "How do I apply for JKM flood disaster relief and bantuan bencana financial assistance in Malaysia?",
  },
  {
    id: "business",
    icon: "🏪",
    label: "Business & Permits",
    description: "SSM registration, SME grants, MyHSB",
    prompt: "What government grants, SSM business registration, and SME support programs are available for small businesses in Malaysia?",
  },
];

// ── Props ────────────────────────────────────────────────────────────────────

interface ScenarioCardsProps {
  onSelect: (prompt: string) => void;
  onStartProfiling?: () => void;
  disabled?: boolean;
}

// ── Component ────────────────────────────────────────────────────────────────

export default function ScenarioCards({ onSelect, onStartProfiling, disabled }: ScenarioCardsProps) {
  return (
    <div className="w-full max-w-md mx-auto mt-2 mb-4 overflow-hidden">
      {/* "Find programs for me" — prominent profiling CTA */}
      {onStartProfiling && (
        <button
          onClick={onStartProfiling}
          disabled={disabled}
          className={cn(
            "w-full mb-4 flex items-center justify-center gap-2.5 px-5 py-3.5 rounded-2xl text-sm font-semibold transition-all duration-200",
            "bg-amber-600 dark:bg-amber-500 text-white dark:text-slate-950",
            "hover:bg-amber-700 dark:hover:bg-amber-400",
            "active:scale-[0.98]",
            "shadow-sm shadow-amber-500/20 dark:shadow-amber-400/20",
            "disabled:opacity-50 disabled:cursor-not-allowed"
          )}
        >
          <span className="text-lg">🔍</span>
          Find programs for me
        </button>
      )}

      {/* Section label */}
      <p className="text-xs font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider mb-3 px-1">
        What can Ara help you with in Malaysia?
      </p>

      {/* Card grid — 2×2 */}
      <div className="grid grid-cols-2 gap-2.5">
        {SCENARIOS.map((scenario) => (
          <button
            key={scenario.id}
            onClick={() => onSelect(scenario.prompt)}
            disabled={disabled}
            className={cn(
              "group relative flex flex-col items-start gap-1.5 p-3.5 rounded-2xl text-left transition-all duration-200",
              "bg-white dark:bg-slate-800/80",
              "border border-slate-200 dark:border-slate-700/80",
              "hover:border-amber-300 dark:hover:border-amber-700",
              "hover:shadow-md hover:shadow-amber-500/5 dark:hover:shadow-amber-400/5",
              "active:scale-[0.97]",
              "disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none disabled:hover:border-slate-200 dark:disabled:hover:border-slate-700/80",
            )}
          >
            {/* Icon */}
            <span className="text-2xl leading-none">{scenario.icon}</span>

            {/* Text */}
            <div>
              <p className="text-sm font-semibold text-slate-800 dark:text-slate-100 leading-tight">
                {scenario.label}
              </p>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-snug mt-0.5">
                {scenario.description}
              </p>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}