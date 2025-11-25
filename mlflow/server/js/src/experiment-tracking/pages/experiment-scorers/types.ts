interface ScheduledScorerBase {
  name: string;
  sampleRate?: number; // Percentage between 0 and 100
  filterString?: string;
  type: 'llm' | 'custom-code';
}

// LLM Template Constants
export const LLM_TEMPLATE = {
  CORRECTNESS: 'Correctness',
  GUIDELINES: 'Guidelines',
  RELEVANCE_TO_QUERY: 'RelevanceToQuery',
  RETRIEVAL_GROUNDEDNESS: 'RetrievalGroundedness',
  RETRIEVAL_RELEVANCE: 'RetrievalRelevance',
  RETRIEVAL_SUFFICIENCY: 'RetrievalSufficiency',
  SAFETY: 'Safety',
  CUSTOM: 'Custom',
} as const;

export type LLMTemplate =
  | 'Correctness'
  | 'Guidelines'
  | 'RelevanceToQuery'
  | 'RetrievalGroundedness'
  | 'RetrievalRelevance'
  | 'RetrievalSufficiency'
  | 'Safety'
  | 'Custom';

export interface LLMScorer extends ScheduledScorerBase {
  type: 'llm';
  llmTemplate?: LLMTemplate;
  guidelines?: string[];
  instructions?: string;
}

export interface CustomCodeScorer extends ScheduledScorerBase {
  type: 'custom-code';
  code: string;
  callSignature: string;
  originalFuncName: string;
}

export type ScheduledScorer = LLMScorer | CustomCodeScorer;

export type ScorerConfig = {
  name: string;
  serialized_scorer: string;
  builtin?: {
    name: string;
  };
  custom?: Record<string, unknown>;
  sample_rate?: number;
  filter_string?: string;
};
