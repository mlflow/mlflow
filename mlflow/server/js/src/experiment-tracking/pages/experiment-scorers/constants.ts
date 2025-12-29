export const COMPONENT_ID_PREFIX = 'mlflow.experiment-scorers';

export const SCORER_FORM_MODE = {
  CREATE: 'create',
  EDIT: 'edit',
  DISPLAY: 'display',
} as const;

export enum ScorerEvaluationScope {
  TRACES = 'traces',
  SESSIONS = 'sessions',
}

export type ScorerFormMode = typeof SCORER_FORM_MODE[keyof typeof SCORER_FORM_MODE];

export const DEFAULT_TRACE_COUNT = 10;

export const ASSESSMENT_NAME_TEMPLATE_MAPPING = {
  Correctness: 'correctness',
  RelevanceToQuery: 'relevance_to_query',
  RetrievalGroundedness: 'groundedness',
  RetrievalSufficiency: 'context_sufficiency',
  Safety: 'harmfulness',
  Guidelines: 'guidelines',
} as const;

export const SCORER_TYPE = {
  LLM: 'llm',
  CUSTOM_CODE: 'custom-code',
} as const;

export type ScorerType = typeof SCORER_TYPE[keyof typeof SCORER_TYPE];

export const BUTTON_VARIANT = {
  RUN: 'run',
  RERUN: 'rerun',
} as const;

export type ButtonVariant = typeof BUTTON_VARIANT[keyof typeof BUTTON_VARIANT];

export const RETRIEVAL_ASSESSMENTS = ['groundedness', 'context_sufficiency'] as const;

export const DEFAULT_LLM_MODEL = 'openai:/gpt-4o-mini';
