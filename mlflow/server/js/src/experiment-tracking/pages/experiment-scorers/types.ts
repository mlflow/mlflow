import {
  ModelTraceLocationMlflowExperiment,
  ModelTraceLocationUcSchema,
} from '@databricks/web-shared/model-trace-explorer';
import { ScorerEvaluationScope } from './constants';

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
  COMPLETENESS = 'Completeness',
  CORRECTNESS = 'Correctness',
  EQUIVALENCE = 'Equivalence',
  EXPECTATIONS_GUIDELINES = 'ExpectationsGuidelines',
  FLUENCY = 'Fluency',
  GUIDELINES = 'Guidelines',
  RELEVANCE_TO_QUERY = 'RelevanceToQuery',
  RETRIEVAL_GROUNDEDNESS = 'RetrievalGroundedness',
  RETRIEVAL_RELEVANCE = 'RetrievalRelevance',
  RETRIEVAL_SUFFICIENCY = 'RetrievalSufficiency',
  SAFETY = 'Safety',
  SUMMARIZATION = 'Summarization',
  TOOL_CALL_CORRECTNESS = 'ToolCallCorrectness',
  TOOL_CALL_EFFICIENCY = 'ToolCallEfficiency',
  CUSTOM = 'Custom',

  // Session-level templates:
  CONVERSATION_COMPLETENESS = 'ConversationCompleteness',
  CONVERSATIONAL_GUIDELINES = 'ConversationalGuidelines',
  CONVERSATIONAL_ROLE_ADHERENCE = 'ConversationalRoleAdherence',
  CONVERSATIONAL_SAFETY = 'ConversationalSafety',
  CONVERSATIONAL_TOOL_CALL_EFFICIENCY = 'ConversationalToolCallEfficiency',
  KNOWLEDGE_RETENTION = 'KnowledgeRetention',
  USER_FRUSTRATION = 'UserFrustration',
}

export const TRACE_LEVEL_LLM_TEMPLATES = [
  LLM_TEMPLATE.COMPLETENESS,
  LLM_TEMPLATE.CORRECTNESS,
  LLM_TEMPLATE.EQUIVALENCE,
  LLM_TEMPLATE.EXPECTATIONS_GUIDELINES,
  LLM_TEMPLATE.FLUENCY,
  LLM_TEMPLATE.GUIDELINES,
  LLM_TEMPLATE.RELEVANCE_TO_QUERY,
  LLM_TEMPLATE.RETRIEVAL_GROUNDEDNESS,
  LLM_TEMPLATE.RETRIEVAL_RELEVANCE,
  LLM_TEMPLATE.RETRIEVAL_SUFFICIENCY,
  LLM_TEMPLATE.SAFETY,
  LLM_TEMPLATE.SUMMARIZATION,
  LLM_TEMPLATE.TOOL_CALL_CORRECTNESS,
  LLM_TEMPLATE.TOOL_CALL_EFFICIENCY,
  LLM_TEMPLATE.CUSTOM,
];

export const SESSION_LEVEL_LLM_TEMPLATES = [
  LLM_TEMPLATE.CONVERSATION_COMPLETENESS,
  LLM_TEMPLATE.CONVERSATIONAL_GUIDELINES,
  LLM_TEMPLATE.CONVERSATIONAL_ROLE_ADHERENCE,
  LLM_TEMPLATE.CONVERSATIONAL_SAFETY,
  LLM_TEMPLATE.CONVERSATIONAL_TOOL_CALL_EFFICIENCY,
  LLM_TEMPLATE.KNOWLEDGE_RETENTION,
  LLM_TEMPLATE.USER_FRUSTRATION,
  LLM_TEMPLATE.CUSTOM,
];

export const TEMPLATES_WITH_GUIDELINES: readonly LLM_TEMPLATE[] = [
  LLM_TEMPLATE.GUIDELINES,
  LLM_TEMPLATE.CONVERSATIONAL_GUIDELINES,
];

export const isGuidelinesTemplate = (template: string | undefined): boolean =>
  template !== undefined && TEMPLATES_WITH_GUIDELINES.includes(template as LLM_TEMPLATE);

export type LLMTemplate =
  | 'Completeness'
  | 'Correctness'
  | 'Equivalence'
  | 'ExpectationsGuidelines'
  | 'Fluency'
  | 'Guidelines'
  | 'RelevanceToQuery'
  | 'RetrievalGroundedness'
  | 'RetrievalRelevance'
  | 'RetrievalSufficiency'
  | 'Safety'
  | 'Summarization'
  | 'ToolCallCorrectness'
  | 'ToolCallEfficiency'
  | 'Custom'
  // Session-level templates:
  | 'ConversationCompleteness'
  | 'ConversationalGuidelines'
  | 'ConversationalRoleAdherence'
  | 'ConversationalSafety'
  | 'ConversationalToolCallEfficiency'
  | 'KnowledgeRetention'
  | 'UserFrustration';

// Primitive types that can be used as dict values or list elements
export type JudgePrimitiveOutputType = 'bool' | 'int' | 'float' | 'str';

// Output type for LLM judges - maps to feedback_value_type in Python SDK
// Primitive types: bool, int, float, str
// Complex types: categorical (Literal), dict, list
// 'default' means no explicit type - let the judge determine automatically
export type JudgeOutputTypeKind = 'default' | JudgePrimitiveOutputType | 'categorical' | 'dict' | 'list';

// Full output type specification
export interface JudgeOutputTypeSpec {
  kind: JudgeOutputTypeKind;
  // For categorical (Literal) type - the list of allowed values
  categoricalOptions?: string[];
  // For dict type - the value type (keys are always strings)
  dictValueType?: JudgePrimitiveOutputType;
  // For list type - the element type
  listElementType?: JudgePrimitiveOutputType;
}

export interface LLMScorer extends ScheduledScorerBase {
  type: 'llm';
  llmTemplate?: LLMTemplate;
  guidelines?: string[];
  instructions?: string;
  model?: string;
  // True if the scorer is an instructions-based LLM scorer that uses instructions_judge_pydantic_data
  // rather than builtin_scorer_pydantic_data.
  is_instructions_judge?: boolean;
  outputType?: JudgeOutputTypeSpec;
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
  serializedScorer?: string;
  evaluationScope?: ScorerEvaluationScope;
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
export type EvaluateTracesParams = (EvaluateChatCompletionsParams | EvaluateChatAssessmentsParams) & {
  saveAssessment?: boolean;
};
