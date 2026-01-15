import { useMemo } from 'react';
import { useIntl } from '@databricks/i18n';
import { LLM_TEMPLATE, SESSION_LEVEL_LLM_TEMPLATES, TRACE_LEVEL_LLM_TEMPLATES } from './types';
import type { FeedbackAssessment } from '@databricks/web-shared/model-trace-explorer';
import { getModelTraceId, isV3ModelTraceInfo } from '@databricks/web-shared/model-trace-explorer';
import { SESSION_TEMPLATE_VARIABLES, TEMPLATE_VARIABLES } from '../../utils/evaluationUtils';
import { ScorerEvaluationScope } from './constants';
import {
  isFeedbackAssessmentInJudgeEvaluationResult,
  isSessionJudgeEvaluationResult,
  JudgeEvaluationResult,
} from './useEvaluateTraces.common';
import { first } from 'lodash';

// Custom hook for template options
export const useTemplateOptions = (scope?: ScorerEvaluationScope) => {
  const intl = useIntl();

  const allTemplateOptions = useMemo(
    () => [
      // Trace-level templates
      {
        value: LLM_TEMPLATE.CORRECTNESS,
        label: intl.formatMessage({ defaultMessage: 'Correctness', description: 'LLM template option' }),
        hint: intl.formatMessage({
          defaultMessage: 'Are the expected facts supported by the response?',
          description: 'Hint for Correctness template',
        }),
      },
      {
        value: LLM_TEMPLATE.GUIDELINES,
        label: intl.formatMessage({ defaultMessage: 'Guidelines', description: 'LLM template option' }),
        hint: intl.formatMessage({
          defaultMessage: 'Does the response follow the provided guidelines?',
          description: 'Hint for Guidelines template',
        }),
      },
      {
        value: LLM_TEMPLATE.RELEVANCE_TO_QUERY,
        label: intl.formatMessage({ defaultMessage: 'Relevance to Query', description: 'LLM template option' }),
        hint: intl.formatMessage({
          defaultMessage: "Does app's response directly address the user's input?",
          description: 'Hint for RelevanceToQuery template',
        }),
      },
      {
        value: LLM_TEMPLATE.RETRIEVAL_GROUNDEDNESS,
        label: intl.formatMessage({ defaultMessage: 'Retrieval Groundedness', description: 'LLM template option' }),
        hint: intl.formatMessage({
          defaultMessage: "Is the app's response grounded in retrieved information?",
          description: 'Hint for RetrievalGroundedness template',
        }),
      },
      {
        value: LLM_TEMPLATE.RETRIEVAL_RELEVANCE,
        label: intl.formatMessage({ defaultMessage: 'Retrieval Relevance', description: 'LLM template option' }),
        hint: intl.formatMessage({
          defaultMessage: "Are retrieved documents relevant to the user's request?",
          description: 'Hint for RetrievalRelevance template',
        }),
      },
      {
        value: LLM_TEMPLATE.RETRIEVAL_SUFFICIENCY,
        label: intl.formatMessage({ defaultMessage: 'Retrieval Sufficiency', description: 'LLM template option' }),
        hint: intl.formatMessage({
          defaultMessage: 'Do retrieved documents contain all necessary information?',
          description: 'Hint for RetrievalSufficiency template',
        }),
      },
      {
        value: LLM_TEMPLATE.SAFETY,
        label: intl.formatMessage({ defaultMessage: 'Safety', description: 'LLM template option' }),
        hint: intl.formatMessage({
          defaultMessage: "Does the app's response avoid harmful or toxic content?",
          description: 'Hint for Safety template',
        }),
      },
      // Session-level templates
      {
        value: LLM_TEMPLATE.CONVERSATION_COMPLETENESS,
        label: intl.formatMessage({ defaultMessage: 'Conversation completeness', description: 'LLM template option' }),
        hint: intl.formatMessage({
          defaultMessage: "Did the conversation fully address the user's request?",
          description: 'Hint for ConversationCompleteness template',
        }),
      },
      {
        value: LLM_TEMPLATE.CONVERSATIONAL_GUIDELINES,
        label: intl.formatMessage({ defaultMessage: 'Conversational guidelines', description: 'LLM template option' }),
        hint: intl.formatMessage({
          defaultMessage: 'Does the assistant follow the provided guidelines throughout the conversation?',
          description: 'Hint for ConversationalGuidelines template',
        }),
      },
      {
        value: LLM_TEMPLATE.KNOWLEDGE_RETENTION,
        label: intl.formatMessage({ defaultMessage: 'Knowledge retention', description: 'LLM template option' }),
        hint: intl.formatMessage({
          defaultMessage: 'Did the assistant remember context from earlier in the conversation?',
          description: 'Hint for KnowledgeRetention template',
        }),
      },
      {
        value: LLM_TEMPLATE.USER_FRUSTRATION,
        label: intl.formatMessage({ defaultMessage: 'User frustration', description: 'LLM template option' }),
        hint: intl.formatMessage({
          defaultMessage: 'Did the conversation avoid causing user frustration?',
          description: 'Hint for UserFrustration template',
        }),
      },
      // Custom template (available for both trace and session level)
      {
        value: LLM_TEMPLATE.CUSTOM,
        label: intl.formatMessage({
          defaultMessage: 'Custom judge',
          description: 'LLM judge option for creating a custom judge',
        }),
        hint: intl.formatMessage({
          defaultMessage: 'Define custom instructions for LLM evaluation',
          description: 'Hint for Custom judge',
        }),
      },
    ],
    [intl],
  );

  const templateOptions = useMemo(() => {
    if (scope === ScorerEvaluationScope.SESSIONS) {
      return allTemplateOptions.filter((option) => SESSION_LEVEL_LLM_TEMPLATES.includes(option.value));
    }

    return allTemplateOptions.filter((option) => TRACE_LEVEL_LLM_TEMPLATES.includes(option.value));
  }, [allTemplateOptions, scope]);

  const displayMap = useMemo(
    () =>
      templateOptions.reduce(
        (map, option) => {
          map[option.value] = option.label;
          return map;
        },
        {} as Record<string, string>,
      ),
    [templateOptions],
  );

  return { templateOptions, displayMap };
};

/**
 * Validates instructions text to ensure only allowed template variables are used
 * and that at least one variable is present.
 *
 * Validation rules:
 * 1. A limited set of reserved variables is allowed based on the scope:
 *    - trace level: inputs, outputs, expectations, trace
 *    - session level: conversation
 * 2. At least one template variable must be present
 * 3. Variable names are case-sensitive and must match exactly
 * 4. {{ trace }} can be combined with {{ inputs }}, {{ outputs }}, or {{ expectations }}
 *    to provide additional context to the agent-based judge
 *
 * @param value - The instructions text to validate
 * @returns true if valid, or an error message string if invalid
 */
export const validateInstructions = (value: string | undefined, scope?: ScorerEvaluationScope): true | string => {
  // Allow empty values - the 'required' rule handles that separately
  if (!value || value.trim() === '') return true;

  const isSessionLevel = scope === ScorerEvaluationScope.SESSIONS;

  const validVariables = isSessionLevel ? SESSION_TEMPLATE_VARIABLES : TEMPLATE_VARIABLES;

  // Extract all template variables in the format {{ variableName }} or {{variableName}}
  const variablePattern = /\{\{\s*(\w+)\s*\}\}/g;
  const matches = [...value.matchAll(variablePattern)];
  const foundVariables = new Set<string>();

  // Check 1: Validate that all variables are from the allowed list
  for (const match of matches) {
    const varName = match[1];
    if (!validVariables.includes(varName)) {
      const validVariablesString = validVariables.map((v) => `{{ ${v} }}`).join(', ');
      return `Invalid variable: {{ ${varName} }}. Only ${validVariablesString} are allowed`;
    }
    foundVariables.add(varName);
  }

  // Check 2: Require at least one template variable
  if (foundVariables.size === 0) {
    if (isSessionLevel) {
      return 'Must contain at least one variable: {{ conversation }}, {{ expectations }}';
    }
    return 'Must contain at least one variable: {{ inputs }}, {{ outputs }}, {{ expectations }}, or {{ trace }}';
  }

  return true;
};

/**
 * Converts a judge evaluation result to a FeedbackAssessment for display
 *
 * @param evaluationResult - The result from evaluating a judge on a trace
 * @param scorerName - The name of the scorer
 * @param index - Optional index for generating unique IDs when creating multiple assessments
 * @returns A FeedbackAssessment object containing the evaluation result or error
 */
export const convertEvaluationResultToAssessment = (
  evaluationResult: JudgeEvaluationResult,
  scorerName: string,
  index?: number,
): FeedbackAssessment => {
  let traceId = 'trace' in evaluationResult && evaluationResult.trace ? getModelTraceId(evaluationResult.trace) : '';

  if (isSessionJudgeEvaluationResult(evaluationResult)) {
    const info = first(evaluationResult.traces)?.info;
    traceId = info && isV3ModelTraceInfo(info) ? info.trace_id : traceId;
  }

  const now = new Date().toISOString();

  // Get the first result from the results array (there should only be one when converting to assessment)
  const firstResult = evaluationResult.results[0];

  // If the evaluation result is already an assessment, return it with trace ID attached and assessment ID generated
  if (firstResult && isFeedbackAssessmentInJudgeEvaluationResult(firstResult)) {
    return { ...firstResult, trace_id: traceId, assessment_id: `${Date.now()}-${index}` };
  }

  return {
    assessment_id: `${Date.now()}`,
    assessment_name: scorerName,
    trace_id: traceId,
    source: {
      source_type: 'LLM_JUDGE',
      source_id: scorerName,
    },
    create_time: now,
    last_update_time: now,
    feedback:
      evaluationResult.error || firstResult?.error
        ? {
            error: {
              error_code: 'INTERNAL_ERROR',
              error_message: evaluationResult.error || firstResult?.error || 'Unknown error',
            },
          }
        : {
            value: firstResult?.result || null,
          },
    rationale: firstResult?.rationale ?? undefined,
    metadata: firstResult?.span_name ? { span_name: firstResult.span_name } : undefined,
  };
};
