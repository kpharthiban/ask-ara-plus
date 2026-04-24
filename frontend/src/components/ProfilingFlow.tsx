"use client";

import { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { RotateCcw } from "lucide-react";
import Image from "next/image";

// ── Profiling step definitions ───────────────────────────────────────────────

interface ProfileOption {
  id: string;
  icon: string;
  label: string;
  value: string;   // sent to backend as part of the profile
}

interface ProfileStep {
  question: string;
  options: ProfileOption[];
}

// AskAra+ is Malaysia-focused. Country is always MY.
// Country picker removed — language selection ≠ knowledge base country.
const STEPS: ProfileStep[] = [
  {
    question: "What's your situation?",
    options: [
      { id: "worker",    icon: "👷",  label: "Worker / Employee",     value: "worker" },
      { id: "immigrant", icon: "🛂",  label: "Migrant / Visa holder", value: "immigrant" },
      { id: "business",  icon: "🏪",  label: "Business owner",        value: "business_owner" },
      { id: "family",    icon: "👨‍👩‍👧", label: "Family / Resident",     value: "family" },
      { id: "disaster",  icon: "🌊",  label: "Disaster affected",      value: "disaster_victim" },
    ],
  },
  {
    question: "What do you need help with?",
    options: [
      { id: "immigration", icon: "🛂", label: "Permit / Visa",   value: "immigration" },
      { id: "financial",   icon: "💰", label: "Financial aid",   value: "financial_aid" },
      { id: "healthcare",  icon: "🏥", label: "Healthcare",      value: "healthcare" },
      { id: "legal",       icon: "📋", label: "Legal rights",    value: "legal_aid" },
      { id: "education",   icon: "🎓", label: "Education",       value: "education" },
    ],
  },
];

// ── Props ────────────────────────────────────────────────────────────────────

interface ProfilingFlowProps {
  /** Called with the composed profile message after all 3 steps complete */
  onComplete: (message: string) => void;
  /** Called when user cancels the flow */
  onCancel: () => void;
  disabled?: boolean;
}

// ── Component ────────────────────────────────────────────────────────────────

export default function ProfilingFlow({ onComplete, onCancel, disabled }: ProfilingFlowProps) {
  // Answers collected so far
  const [answers, setAnswers] = useState<ProfileOption[]>([]);
  const currentStep = answers.length;
  const isDone = currentStep >= STEPS.length;

  const handleSelect = useCallback(
    (option: ProfileOption) => {
      const newAnswers = [...answers, option];
      setAnswers(newAnswers);

      // If all 2 steps done → compose message and fire
      // Country is always MY — AskAra+ is Malaysia-focused.
      if (newAnswers.length === STEPS.length) {
        const [situation, need] = newAnswers;

        const message =
          `Find government programs for me. ` +
          `I am in Malaysia (MY). ` +
          `My situation: ${situation.value}. ` +
          `I need help with: ${need.value}.`;

        // Small delay so user sees their last selection before chat takes over
        setTimeout(() => onComplete(message), 300);
      }
    },
    [answers, onComplete]
  );

  const handleReset = useCallback(() => {
    setAnswers([]);
  }, []);

  if (isDone) return null;

  const step = STEPS[currentStep];

  return (
    <div className="flex w-full min-w-0 mb-6 justify-start">
      {/* Ara avatar */}
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-amber-100 dark:bg-amber-900/50 mr-4 mt-0.5 border border-amber-200/50 dark:border-amber-800/50 shadow-sm transition-colors duration-500 overflow-hidden">
        <Image src="/icons/cat.png" alt="Ara" width={20} height={20} className="object-contain" />
      </div>

      <div className="max-w-[85%] md:max-w-[75%] min-w-0 overflow-hidden">
        {/* Show previous answers as recap */}
        {answers.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-3">
            {answers.map((ans, i) => (
              <span
                key={i}
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 border border-amber-200/50 dark:border-amber-800/50"
              >
                <span>{ans.icon}</span>
                <span>{ans.label}</span>
              </span>
            ))}
          </div>
        )}

        {/* Current question */}
        <p className="text-[16px] leading-relaxed text-slate-800 dark:text-slate-200 font-medium mb-3">
          {step.question}
        </p>

        {/* Quick-reply buttons */}
        <div className="flex flex-wrap gap-2">
          {step.options.map((option) => (
            <button
              key={option.id}
              onClick={() => handleSelect(option)}
              disabled={disabled}
              className={cn(
                "inline-flex items-center gap-1.5 px-4 py-2.5 rounded-2xl text-sm font-medium transition-all duration-200",
                "bg-white dark:bg-slate-800",
                "border border-slate-200 dark:border-slate-700",
                "text-slate-700 dark:text-slate-200",
                "hover:border-amber-400 dark:hover:border-amber-500",
                "hover:bg-teal-50 dark:hover:bg-teal-900/30",
                "hover:text-amber-700 dark:hover:text-amber-300",
                "active:scale-[0.96]",
                "disabled:opacity-50 disabled:cursor-not-allowed"
              )}
            >
              <span className="text-lg leading-none">{option.icon}</span>
              <span>{option.label}</span>
            </button>
          ))}
        </div>

        {/* Reset / Cancel row */}
        <div className="flex items-center gap-3 mt-3">
          {answers.length > 0 && (
            <button
              onClick={handleReset}
              className="inline-flex items-center gap-1 text-xs text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
            >
              <RotateCcw className="h-3 w-3" />
              Start over
            </button>
          )}
          <button
            onClick={onCancel}
            className="text-xs text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}