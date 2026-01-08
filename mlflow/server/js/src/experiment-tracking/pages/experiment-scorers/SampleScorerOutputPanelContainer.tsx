import React, { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import type { Control } from 'react-hook-form';
import { useWatch, useFormState, useFormContext } from 'react-hook-form';
import { useIntl } from '@databricks/i18n';
import { type ScorerFormData } from './utils/scorerTransformUtils';
import { useEvaluateTraces } from './useEvaluateTraces';
import SampleScorerOutputPanelRenderer from './SampleScorerOutputPanelRenderer';
import { convertEvaluationResultToAssessment } from './llmScorerUtils';
import { extractTemplateVariables } from '../../utils/evaluationUtils';
import { DEFAULT_TRACE_COUNT, ASSESSMENT_NAME_TEMPLATE_MAPPING, ScorerEvaluationScope, SCORER_TYPE } from './constants';
import { EvaluateTracesParams, LLM_TEMPLATE } from './types';
import { coerceToEnum } from '../../../shared/web-shared/utils';
import { useGetSerializedScorerFromForm } from './useGetSerializedScorerFromForm';
import { JudgeEvaluationResult } from './useEvaluateTraces.common';
import { isEvaluatingSessionsInScorersEnabled } from '../../../common/utils/FeatureUtils';

interface SampleScorerOutputPanelContainerProps {
  control: Control<ScorerFormData>;
  experimentId: string;
  onScorerFinished?: () => void;
  isSessionLevelScorer?: boolean;
}

const SampleScorerOutputPanelContainer: React.FC<SampleScorerOutputPanelContainerProps> = ({
  control,
  experimentId,
  onScorerFinished,
  isSessionLevelScorer,
}) => {
  const intl = useIntl();
  const judgeInstructions = useWatch({ control, name: 'instructions' });
  const scorerName = useWatch({ control, name: 'name' });
  const llmTemplate = useWatch({ control, name: 'llmTemplate' });
  const guidelines = useWatch({ control, name: 'guidelines' });
  const modelValue = useWatch({ control, name: 'model' });
  const { resetField } = useFormContext<ScorerFormData>();
  const { errors } = useFormState({ control });
  const evaluationScopeFormValue = useWatch({ control, name: 'evaluationScope' });
  const evaluationScope = coerceToEnum(ScorerEvaluationScope, evaluationScopeFormValue, ScorerEvaluationScope.TRACES);

  const getSerializedScorerFromForm = useGetSerializedScorerFromForm();

  const [itemsToEvaluate, setItemsToEvaluate] = useState<Pick<EvaluateTracesParams, 'itemCount' | 'itemIds'>>({
    itemCount: DEFAULT_TRACE_COUNT,
    itemIds: [],
  });

  const [evaluateTraces, { data, isLoading, error, reset }] = useEvaluateTraces({
    onScorerFinished,
  });

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

  // Reset evaluation config when switching evaluation scope
  useEffect(() => {
    reset();
    setItemsToEvaluate({
      itemCount: DEFAULT_TRACE_COUNT,
      itemIds: [],
    });
    resetField('instructions');
    resetField('llmTemplate');
  }, [evaluationScope, reset, resetField]);

  // Handle the "Run scorer" button click
  const handleRunScorer = useCallback(async () => {
    // Validate inputs based on mode
    if (isCustomMode ? !judgeInstructions : !llmTemplate) {
      return;
    }

    const serializedScorer = getSerializedScorerFromForm();

    // Reset to first trace when running scorer
    setCurrentTraceIndex(0);

    try {
      // Prepare evaluation parameters based on mode
      const evaluationParams = isCustomMode
        ? {
            itemCount: itemsToEvaluate.itemCount,
            itemIds: itemsToEvaluate.itemIds,
            locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' as const }],
            judgeInstructions: judgeInstructions || '',
            experimentId,
            serializedScorer,
            evaluationScope,
          }
        : {
            itemCount: itemsToEvaluate.itemCount,
            itemIds: itemsToEvaluate.itemIds,
            locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' as const }],
            requestedAssessments: [
              {
                assessment_name:
                  ASSESSMENT_NAME_TEMPLATE_MAPPING[llmTemplate as keyof typeof ASSESSMENT_NAME_TEMPLATE_MAPPING],
              },
            ],
            experimentId,
            guidelines: guidelines ? [guidelines] : undefined,
            serializedScorer,
            evaluationScope,
          };

      await evaluateTraces(evaluationParams);
    } catch (error) {
      // Error is already handled by the hook's error state
    }
  }, [
    isCustomMode,
    judgeInstructions,
    llmTemplate,
    guidelines,
    itemsToEvaluate,
    evaluationScope,
    evaluateTraces,
    experimentId,
    getSerializedScorerFromForm,
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
        } as JudgeEvaluationResult,
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
  const isGuidelinesTemplate =
    llmTemplate === LLM_TEMPLATE.GUIDELINES || llmTemplate === LLM_TEMPLATE.CONVERSATIONAL_GUIDELINES;
  const hasEmptyGuidelines = isGuidelinesTemplate && (!guidelines || !guidelines.trim());

  // Determine tooltip message based on why the button is disabled
  const runScorerDisabledReason = useMemo(() => {
    if (!modelValue) {
      return intl.formatMessage({
        defaultMessage: 'Please select a model to run the judge',
        description: 'Tooltip message when model is not selected',
      });
    }

    if (!isEvaluatingSessionsInScorersEnabled() && isSessionLevelScorer) {
      return intl.formatMessage({
        defaultMessage: 'Session-level scorers cannot be run on individual traces',
        description: 'Tooltip message when scorer is session-level',
      });
    }

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
      if (hasEmptyGuidelines) {
        return intl.formatMessage({
          defaultMessage: 'Guidelines should not be empty',
          description: 'Tooltip message when guidelines are empty',
        });
      }
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
    isSessionLevelScorer,
    modelValue,
    isCustomMode,
    judgeInstructions,
    hasInstructionsError,
    hasTraceVariable,
    hasNameError,
    isRetrievalRelevance,
    hasEmptyGuidelines,
    intl,
  ]);

  const isRunScorerDisabled = Boolean(runScorerDisabledReason);

  return (
    <SampleScorerOutputPanelRenderer
      isLoading={isLoading}
      isRunScorerDisabled={isRunScorerDisabled}
      runScorerDisabledTooltip={runScorerDisabledReason}
      error={error}
      currentEvalResultIndex={currentTraceIndex}
      currentEvalResult={currentEvalResult}
      assessments={assessments}
      handleRunScorer={handleRunScorer}
      handlePrevious={handlePrevious}
      handleNext={handleNext}
      totalTraces={data?.length ?? 0}
      itemsToEvaluate={itemsToEvaluate}
      onItemsToEvaluateChange={setItemsToEvaluate}
    />
  );
};

export default SampleScorerOutputPanelContainer;
