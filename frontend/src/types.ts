export enum AppState {
  IDLE = 'IDLE',
  PROCESSING = 'PROCESSING',
  SUCCESS = 'SUCCESS',
  ERROR = 'ERROR',
}

export enum SummaryLength {
  SHORT = 'short',
  MEDIUM = 'medium',
  LONG = 'long',
}

export enum VoiceOption {
  MALE = 'male',
  FEMALE = 'female',
}

export interface AnswerSet {
  [SummaryLength.SHORT]: string;
  [SummaryLength.MEDIUM]: string;
  [SummaryLength.LONG]: string;
}

export interface Claim {
  claim: string;
  verdict: 'SUPPORTS' | 'REFUTES' | 'NEUTRAL';
  confidence: number;
  explanation: string;
}

export interface QAPair {
  id: string;
  question: string;
  answers: AnswerSet;
  claims: Claim[];
}

export interface SummarySet {
  [SummaryLength.SHORT]: string;
  [SummaryLength.MEDIUM]: string;
  [SummaryLength.LONG]: string;
}

export interface SummaryResult {
  title: string;
  thumbnailUrl?: string;
  overallSummary: SummarySet;
  qaPairs: QAPair[];
  cached?: boolean;
}

// Types for the Episode Sequencing feature
export interface EpisodeSegment {
  id: string;
  title: string;
  summary: string;
  sourcePodcast: number;
  keyConcept: string;
}

export interface EpisodeSequenceResult {
  sequenceTitle: string;
  sequenceIntroduction: string;
  segments: EpisodeSegment[];
}

export interface HistoryItem {
  uploadid: number;
  source: string;
  title: string;
  thumbnail_url?: string;
  created_at: string;
}
