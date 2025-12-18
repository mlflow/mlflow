import { useMemo } from 'react';
import { useIntl } from '@databricks/i18n';
import { LLM_TEMPLATE } from './types';
import type { FeedbackAssessment } from '@databricks/web-shared/model-trace-explorer';
import { getModelTraceId } from '@databricks/web-shared/model-trace-explorer';
import type { JudgeEvaluationResult } from './useEvaluateTraces';
import { TEMPLATE_VARIABLES } from '../../utils/evaluationUtils';

// Custom hook for template options
export const useTemplateOptions = () => {
  const intl = useIntl();

  const templateOptions = useMemo(
    () => [
      {
        value: LLM_TEMPLATE.CORRECTNESS,
        label: intl.formatMessage({ defaultMessage: 'Correctness', description: 'LLM template option' }),
        hint: intl.formatMessage({
          defaultMessage: 'Are the expected facts supported by the response?',
          description: 'Hint for Correctness template',
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
      {
        value: LLM_TEMPLATE.CUSTOM,
        label: intl.formatMessage({
          defaultMessage: 'Create custom LLM template',
          description: 'LLM template option',
        }),
        hint: intl.formatMessage({
          defaultMessage: 'Define custom instructions for LLM evaluation',
          description: 'Hint for Custom template',
        }),
      },
    ],
    [intl],
  );

  const displayMap = useMemo(
    () =>
      templateOptions.reduce((map, option) => {
        map[option.value] = option.label;
        return map;
      }, {} as Record<string, string>),
    [templateOptions],
  );

  return { templateOptions, displayMap };
};

/**
 * Validates instructions text to ensure only allowed template variables are used
 * and that at least one variable is present.
 *
 * Validation rules:
 * 1. Only 4 reserved variables are allowed: inputs, outputs, expectations, trace
 * 2. At least one template variable must be present
 * 3. Variable names are case-sensitive and must match exactly
 * 4. {{ trace }} can be combined with {{ inputs }}, {{ outputs }}, or {{ expectations }}
 *    to provide additional context to the agent-based judge
 *
 * @param value - The instructions text to validate
 * @returns true if valid, or an error message string if invalid
 */
export const validateInstructions = (value: string | undefined): true | string => {
  // Allow empty values - the 'required' rule handles that separately
  if (!value || value.trim() === '') return true;

  // Extract all template variables in the format {{ variableName }} or {{variableName}}
  const variablePattern = /\{\{\s*(\w+)\s*\}\}/g;
  const matches = [...value.matchAll(variablePattern)];
  const foundVariables = new Set<string>();

  // Check 1: Validate that all variables are from the allowed list
  for (const match of matches) {
    const varName = match[1];
    if (!TEMPLATE_VARIABLES.includes(varName)) {
      return `Invalid variable: {{ ${varName} }}. Only {{ inputs }}, {{ outputs }}, {{ expectations }}, {{ trace }} are allowed`;
    }
    foundVariables.add(varName);
  }

  // Check 2: Require at least one template variable
  if (foundVariables.size === 0) {
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
  const traceId = evaluationResult.trace ? getModelTraceId(evaluationResult.trace) : '';
  const now = new Date().toISOString();

  // Get the first result from the results array (there should only be one when converting to assessment)
  const firstResult = evaluationResult.results[0];

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
