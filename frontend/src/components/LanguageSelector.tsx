"use client";

import { cn } from "@/lib/utils";

// ── Language options ─────────────────────────────────────────────────────────
// `code`    = BCP-47 tag sent to backend & used by Web Speech API
// `country` = always "MY" — AskAra+ knowledge base is Malaysia-only.
//             Language selection drives response language, NOT the KB country.

export interface LanguageOption {
  id: string;          // internal key
  flag: string;        // emoji flag
  label: string;       // native-script label
  code: string;        // BCP-47 language tag
  country: string;     // always "MY" — knowledge base is Malaysia-only
}

export const LANGUAGES: LanguageOption[] = [
  { id: "auto", flag: "🌐", label: "Auto",             code: "",       country: "MY" },
  { id: "ms",   flag: "🇲🇾", label: "Bahasa Melayu",   code: "ms-MY",  country: "MY" },
  { id: "en",   flag: "🇬🇧", label: "English",          code: "en-MY",  country: "MY" },
  { id: "id",   flag: "🇮🇩", label: "Bahasa Indonesia", code: "id-ID",  country: "MY" },
  { id: "fil",  flag: "🇵🇭", label: "Filipino",         code: "fil-PH", country: "MY" },
  { id: "th",   flag: "🇹🇭", label: "ภาษาไทย",          code: "th-TH",  country: "MY" },
];

// ── Props ────────────────────────────────────────────────────────────────────

interface LanguageSelectorProps {
  selected: string;                       // currently selected id
  onChange: (lang: LanguageOption) => void;
}

// ── Component ────────────────────────────────────────────────────────────────

export default function LanguageSelector({ selected, onChange }: LanguageSelectorProps) {
  return (
    <div className="flex items-center gap-1.5 px-1 py-1.5 w-full overflow-x-auto scrollbar-none">
      {LANGUAGES.map((lang) => {
        const isActive = selected === lang.id;

        return (
          <button
            key={lang.id}
            type="button"
            onClick={() => onChange(lang)}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold whitespace-nowrap transition-all duration-200 shrink-0",
              "border focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500 focus-visible:ring-offset-1",
              isActive
                ? "bg-amber-50 dark:bg-amber-900/40 border-amber-300 dark:border-amber-600 text-amber-700 dark:text-amber-300 shadow-sm"
                : "bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400 hover:border-slate-300 dark:hover:border-slate-600 hover:text-slate-700 dark:hover:text-slate-300"
            )}
            aria-pressed={isActive}
            aria-label={`Select language: ${lang.label}`}
          >
            <span className="text-sm leading-none">{lang.flag}</span>
            <span>{lang.label}</span>
          </button>
        );
      })}
    </div>
  );
}