import {
  ModelTraceLocationMlflowExperiment,
  ModelTraceLocationUcSchema,
} from '@databricks/web-shared/model-trace-explorer';

interface ScheduledScorerBase {
  name: string;
  sampleRate?: number; // Percentage between 0 and 100
  filterString?: string;
  type: 'llm' | 'custom-code';
  version?: number;
  // Whether the UI disables monitoring for this scorer. If disabled, the UI
  // will not show the form fields for monitoring (sample rate, filter string, etc.)
  disableMonitoring?: boolean;
  isSessionLevelScorer?: boolean;
}

// LLM Template Constants
export enum LLM_TEMPLATE {
  CORRECTNESS = 'Correctness',
  GUIDELINES = 'Guidelines',
  RELEVANCE_TO_QUERY = 'RelevanceToQuery',
  RETRIEVAL_GROUNDEDNESS = 'RetrievalGroundedness',
  RETRIEVAL_RELEVANCE = 'RetrievalRelevance',
  RETRIEVAL_SUFFICIENCY = 'RetrievalSufficiency',
  SAFETY = 'Safety',
  CUSTOM = 'Custom',

  // Session-level templates:
  CONVERSATION_COMPLETENESS = 'ConversationCompleteness',
  KNOWLEDGE_RETENTION = 'KnowledgeRetention',
  USER_FRUSTRATION = 'UserFrustration',
}

export const TRACE_LEVEL_LLM_TEMPLATES = [
  LLM_TEMPLATE.CORRECTNESS,
  LLM_TEMPLATE.GUIDELINES,
  LLM_TEMPLATE.RELEVANCE_TO_QUERY,
  LLM_TEMPLATE.RETRIEVAL_GROUNDEDNESS,
  LLM_TEMPLATE.RETRIEVAL_RELEVANCE,
  LLM_TEMPLATE.RETRIEVAL_SUFFICIENCY,
  LLM_TEMPLATE.SAFETY,
  LLM_TEMPLATE.CUSTOM,
];

export const SESSION_LEVEL_LLM_TEMPLATES = [
  LLM_TEMPLATE.CONVERSATION_COMPLETENESS,
  LLM_TEMPLATE.KNOWLEDGE_RETENTION,
  LLM_TEMPLATE.USER_FRUSTRATION,
  LLM_TEMPLATE.CUSTOM,
];

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
  model?: string;
  // True if the scorer is an instructions-based LLM scorer that uses instructions_judge_pydantic_data
  // rather than builtin_scorer_pydantic_data.
  is_instructions_judge?: boolean;
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
  scorer_version?: number;
  is_session_level_scorer?: boolean;
};

interface EvaluateChatParamsBase {
  // Number of items to evaluate. This can be the number of traces or sessions.
  itemCount?: number;
  // Explicit list of item IDs to evaluate. Can be either trace IDs or session IDs. This is used to override the itemCount.
  itemIds?: string[];
  locations: (ModelTraceLocationMlflowExperiment | ModelTraceLocationUcSchema)[];
  experimentId: string;
}

/**
 * Parameters for evaluating traces with custom LLM judges
 */
export interface EvaluateChatCompletionsParams extends EvaluateChatParamsBase {
  judgeInstructions: string;
}

/**
 * Parameters for evaluating traces with built-in judges
 */
export interface EvaluateChatAssessmentsParams extends EvaluateChatParamsBase {
  requestedAssessments: Array<{
    assessment_name: string;
    assessment_examples?: any[];
  }>;
  guidelines?: string[];
}

/**
 * Union type for all trace evaluation parameters
 */
export type EvaluateTracesParams = EvaluateChatCompletionsParams | EvaluateChatAssessmentsParams;
