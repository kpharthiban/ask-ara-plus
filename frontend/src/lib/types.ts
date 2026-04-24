export type Sender = "user" | "ara";

export type MessageContentType =
  | "text"
  | "step_cards"
  | "recommendations"
  | "image"
  | "voice_transcript";

export interface Source {
  title: string;
  url?: string;
  relevance?: number;
  source_agency?: string;
  country?: string;
}

// ── Step Cards ────────────────────────────────────────────────────────────

export interface StepCardAction {
  type: "navigate" | "call" | "share" | "link" | "none";
  label?: string;
  url?: string;
  phone?: string;
  lat?: number;
  lng?: number;
}

export interface StepCard {
  step: number;
  total: number;
  title: string;
  body?: string;
  icon?: string;           // emoji icon for the card
  location?: string;
  hours?: string;
  deadline?: string;
  amount?: string;          // e.g. "RM500"
  checklist?: string[];     // e.g. ["IC (MyKad)", "Bukti alamat"]
  action?: StepCardAction;
}

export interface StepCardsPayload {
  type: "step_cards";
  cards: StepCard[];
  summary?: string;        // optional intro text above the cards
}

// ── Recommendations ───────────────────────────────────────────────────────

export interface Recommendation {
  id: string;
  title: string;
  description: string;
  icon?: string;
  country?: string;
  eligibility?: string;
  action?: StepCardAction;
  tags?: string[];
}

export interface RecommendationsPayload {
  type: "recommendations";
  title?: string;
  items: Recommendation[];
}

// ── Structured content union ──────────────────────────────────────────────

export type StructuredContent = StepCardsPayload | RecommendationsPayload;

// ── Message ───────────────────────────────────────────────────────────────

export interface Message {
  id: string;
  sender: Sender;
  content: string;                      // text content (or raw JSON string)
  contentType: MessageContentType;
  timestamp: Date;
  isLoading?: boolean;
  isStreaming?: boolean;
  sources?: Source[];
  toolCalls?: string[];
  structured?: StructuredContent;       // parsed structured payload
  /**
   * Base64 data-URI of an image attached to this message.
   * Only present on user messages that include an uploaded image.
   * Used to render the thumbnail in the chat bubble.
   */
  imagePreview?: string;
}