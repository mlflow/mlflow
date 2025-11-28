import React, { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import type { Control } from 'react-hook-form';
import { useWatch, useFormState } from 'react-hook-form';
import { useIntl } from '@databricks/i18n';
import type { ScorerFormData } from './utils/scorerTransformUtils';
import { useEvaluateTraces } from './useEvaluateTraces';
import SampleScorerOutputPanelRenderer from './SampleScorerOutputPanelRenderer';
import { convertEvaluationResultToAssessment } from './llmScorerUtils';
import { extractTemplateVariables } from '../../utils/evaluationUtils';
import { DEFAULT_TRACE_COUNT, ASSESSMENT_NAME_TEMPLATE_MAPPING } from './constants';
import { LLM_TEMPLATE } from './types';

interface SampleScorerOutputPanelContainerProps {
  control: Control<ScorerFormData>;
  experimentId: string;
  onScorerFinished?: () => void;
}

const SampleScorerOutputPanelContainer: React.FC<SampleScorerOutputPanelContainerProps> = ({
  control,
  experimentId,
  onScorerFinished,
}) => {
  const intl = useIntl();
  const judgeInstructions = useWatch({ control, name: 'instructions' });
  const scorerName = useWatch({ control, name: 'name' });
  const llmTemplate = useWatch({ control, name: 'llmTemplate' });
  const guidelines = useWatch({ control, name: 'guidelines' });
  const scorerType = useWatch({ control, name: 'scorerType' });
  const { errors } = useFormState({ control });

  const [tracesCount, setTracesCount] = useState(DEFAULT_TRACE_COUNT);
  const [evaluateTraces, { data, isLoading, error, reset }] = useEvaluateTraces();

  // Carousel state for navigating through traces
  const [currentTraceIndex, setCurrentTraceIndex] = useState(0);

  // Request ID pattern to handle stale results
  const requestIdRef = useRef(0);

  // Determine if we're in custom or built-in judge mode
  const isCustomMode = llmTemplate === LLM_TEMPLATE.CUSTOM;

  // Reset results when switching modes or templates
  useEffect(() => {
    reset();
    setCurrentTraceIndex(0);
  }, [llmTemplate, reset]);

  // Handle the "Run scorer" button click
  const handleRunScorer = useCallback(async () => {
    // Validate inputs based on mode
    if (isCustomMode ? !judgeInstructions : !llmTemplate) {
      return;
    }

    // Increment request ID and save for this request
    requestIdRef.current += 1;
    const thisRequestId = requestIdRef.current;

    // Reset to first trace when running scorer
    setCurrentTraceIndex(0);

    try {
      // Prepare evaluation parameters based on mode
      const evaluationParams = isCustomMode
        ? {
            traceCount: tracesCount,
            locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' as const }],
            judgeInstructions: judgeInstructions || '',
            experimentId,
          }
        : {
            traceCount: tracesCount,
            locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' as const }],
            requestedAssessments: [
              {
                assessment_name:
                  ASSESSMENT_NAME_TEMPLATE_MAPPING[llmTemplate as keyof typeof ASSESSMENT_NAME_TEMPLATE_MAPPING],
              },
            ],
            experimentId,
            guidelines: guidelines ? [guidelines] : undefined,
          };

      const results = await evaluateTraces(evaluationParams);

      // Check if results are still current (user hasn't changed settings)
      if (thisRequestId === requestIdRef.current) {
        // Call onScorerFinished after successful evaluation
        if (results && results.length > 0) {
          onScorerFinished?.();
        }
      }
    } catch (error) {
      // Error is already handled by the hook's error state
    }
  }, [
    isCustomMode,
    judgeInstructions,
    llmTemplate,
    guidelines,
    tracesCount,
    evaluateTraces,
    experimentId,
    onScorerFinished,
  ]);

  // Navigation handlers
  const handlePrevious = () => {
    setCurrentTraceIndex((prev) => Math.max(0, prev - 1));
  };

  const handleNext = () => {
    if (data) {
      setCurrentTraceIndex((prev) => Math.min(data.length - 1, prev + 1));
    }
  };

  // Get current evaluation result from data
  const currentEvalResult = data?.[currentTraceIndex];

  // Convert judge evaluation result to assessments for display
  const assessments = useMemo(() => {
    if (!currentEvalResult || !llmTemplate) {
      return undefined;
    }

    const baseName = scorerName || llmTemplate;

    // Custom judges or built-in judges with no results: single assessment
    if (isCustomMode || currentEvalResult.results.length === 0) {
      const assessment = convertEvaluationResultToAssessment(currentEvalResult, baseName);
      return [assessment];
    }

    // Built-in judges with results: map over each result
    return currentEvalResult.results.map((result, index) => {
      return convertEvaluationResultToAssessment(
        {
          ...currentEvalResult,
          results: [result],
        },
        baseName,
        index,
      );
    });
  }, [currentEvalResult, isCustomMode, llmTemplate, scorerName]);

  // Check if instructions contain {{trace}} variable (custom judges only)
  const hasTraceVariable = useMemo(() => {
    if (!isCustomMode || !judgeInstructions) return false;
    const templateVariables = extractTemplateVariables(judgeInstructions);
    return templateVariables.includes('trace');
  }, [isCustomMode, judgeInstructions]);

  // Determine if run scorer button should be disabled
  const hasNameError = Boolean((errors as any).name?.message);
  const hasInstructionsError = Boolean((errors as any).instructions?.message);
  const isRetrievalRelevance = llmTemplate === LLM_TEMPLATE.RETRIEVAL_RELEVANCE;

  const isRunScorerDisabled = isCustomMode
    ? !judgeInstructions || hasInstructionsError || hasTraceVariable
    : hasNameError || isRetrievalRelevance;

  // Determine tooltip message based on why the button is disabled
  const runScorerDisabledTooltip = useMemo(() => {
    if (isCustomMode) {
      // Custom judge mode
      if (!judgeInstructions) {
        return intl.formatMessage({
          defaultMessage: 'Please enter instructions to run the judge',
          description: 'Tooltip message when instructions are missing',
        });
      }
      if (hasInstructionsError) {
        return intl.formatMessage({
          defaultMessage: 'Please fix the validation errors in the instructions',
          description: 'Tooltip message when instructions have validation errors',
        });
      }
      if (hasTraceVariable) {
        return intl.formatMessage({
          defaultMessage: 'The trace variable is not supported when running the judge on a sample of traces',
          description: 'Tooltip message when instructions contain trace variable',
        });
      }
    } else {
      // Built-in judge mode
      if (isRetrievalRelevance) {
        return intl.formatMessage({
          defaultMessage: 'Retrieval Relevance is not yet supported for sample judge output',
          description: 'Tooltip message when retrieval relevance template is selected',
        });
      }
      if (hasNameError) {
        return intl.formatMessage({
          defaultMessage: 'Please fix the validation errors',
          description: 'Tooltip message when there are validation errors',
        });
      }
    }
    return undefined;
  }, [
    isCustomMode,
    judgeInstructions,
    hasInstructionsError,
    hasTraceVariable,
    hasNameError,
    isRetrievalRelevance,
    intl,
  ]);

  return (
    <SampleScorerOutputPanelRenderer
      isLoading={isLoading}
      isRunScorerDisabled={isRunScorerDisabled}
      runScorerDisabledTooltip={runScorerDisabledTooltip}
      error={error}
      currentTraceIndex={currentTraceIndex}
      currentTrace={currentEvalResult?.trace ?? undefined}
      assessments={assessments}
      handleRunScorer={handleRunScorer}
      handlePrevious={handlePrevious}
      handleNext={handleNext}
      totalTraces={data?.length ?? 0}
      tracesCount={tracesCount}
      onTracesCountChange={setTracesCount}
    />
  );
};

export default SampleScorerOutputPanelContainer;
