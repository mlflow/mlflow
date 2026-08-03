import React, { useState, useMemo, useCallback, useEffect } from 'react';
import { useSqlWarehouseContextSafe } from '../experiment-page-tabs/SqlWarehouseContext';
import { createTraceLocationForExperiment } from '@databricks/web-shared/genai-traces-table';
import type { Control } from 'react-hook-form';
import { useWatch, useFormState } from 'react-hook-form';
import { useIntl } from '@databricks/i18n';
import { type ScorerFormData } from './utils/scorerTransformUtils';
import { useEvaluateTraces } from './useEvaluateTraces';
import SampleScorerOutputPanelRenderer from './SampleScorerOutputPanelRenderer';
import { convertEvaluationResultToAssessment } from './llmScorerUtils';
import { extractTemplateVariables } from '../../utils/evaluationUtils';
import { ASSESSMENT_NAME_TEMPLATE_MAPPING, ScorerEvaluationScope } from './constants';

import { LLM_TEMPLATE, isGuidelinesTemplate } from './types';
import { coerceToEnum } from '../../../shared/web-shared/utils';
import { useGetSerializedScorerFromForm } from './useGetSerializedScorerFromForm';
import type { JudgeEvaluationResult } from './useEvaluateTraces.common';
import {
  isEvaluatingSessionsInScorersEnabled,
  isRunningAgenticJudgesEnabled,
  isRunningAllScorerTemplatesEnabled,
  isScorerModelSelectionEnabled,
  shouldSupportRunningDatabricksProviderJudgesFromUI,
} from '../../../common/utils/FeatureUtils';
import { getModelProvider, ModelProvider } from '../../../gateway/utils/gatewayUtils';

interface SampleScorerOutputPanelContainerProps {
  control: Control<ScorerFormData>;
  experimentId: string;
  onScorerFinished?: () => void;
  isSessionLevelScorer?: boolean;
  selectedItemIds: string[];
  onSelectedItemIdsChange: (itemIds: string[]) => void;
}

const SampleScorerOutputPanelContainer: React.FC<SampleScorerOutputPanelContainerProps> = ({
  control,
  experimentId,
  onScorerFinished,
  isSessionLevelScorer,
  selectedItemIds,
  onSelectedItemIdsChange,
}) => {
  const intl = useIntl();
  const {
    warehouseId: selectedWarehouseId,
    setWarehouseId: setSelectedWarehouseId,
    traceSearchLocations = [createTraceLocationForExperiment(experimentId)],
    hasV4Location,
  } = useSqlWarehouseContextSafe() ?? {};
  const showWarehouseSelector = hasV4Location;
  const judgeInstructions = useWatch({ control, name: 'instructions' });
  const scorerName = useWatch({ control, name: 'name' });
  const llmTemplate = useWatch({ control, name: 'llmTemplate' });
  const guidelines = useWatch({ control, name: 'guidelines' });
  const modelValue = useWatch({ control, name: 'model' });
  const { errors } = useFormState({ control });
  const isInstructionsJudge = useWatch({ control, name: 'isInstructionsJudge' });
  const evaluationScopeFormValue = useWatch({ control, name: 'evaluationScope' });
  const evaluationScope = coerceToEnum(ScorerEvaluationScope, evaluationScopeFormValue, ScorerEvaluationScope.TRACES);

  const getSerializedScorerFromForm = useGetSerializedScorerFromForm();

  const [evaluateTraces, { latestEvaluation: data, isLoading, error, reset }] = useEvaluateTraces({
    onScorerFinished,
  });

  // Carousel state for navigating through traces
  const [currentTraceIndex, setCurrentTraceIndex] = useState(0);

  // Reset results when switching modes or templates
  useEffect(() => {
    reset();
    setCurrentTraceIndex(0);
  }, [llmTemplate, reset]);

  // Handle the "Run scorer" button click
  const handleRunScorer = useCallback(async () => {
    // Validate inputs based on mode
    if (isInstructionsJudge ? !judgeInstructions : !llmTemplate) {
      return;
    }

    const serializedScorer = getSerializedScorerFromForm();

    // Reset to first trace when running scorer
    setCurrentTraceIndex(0);

    try {
      // Prepare evaluation parameters based on mode
      const evaluationParams = isInstructionsJudge
        ? {
            itemIds: selectedItemIds,
            locations: traceSearchLocations,
            judgeInstructions: judgeInstructions || '',
            experimentId,
            serializedScorer,
            evaluationScope,
          }
        : {
            itemIds: selectedItemIds,
            locations: traceSearchLocations,
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
    // prettier-ignore
  }, [
    isInstructionsJudge,
    judgeInstructions,
    llmTemplate,
    guidelines,
    selectedItemIds,
    evaluationScope,
    evaluateTraces,
    experimentId,
    getSerializedScorerFromForm,
    traceSearchLocations,
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
    if (isInstructionsJudge || currentEvalResult.results.length === 0) {
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
  }, [currentEvalResult, isInstructionsJudge, llmTemplate, scorerName]);

  // Check if instructions contain {{trace}} variable (only supported when agentic judges are enabled)
  const isTraceVariableBlocked = useMemo(() => {
    if (isRunningAgenticJudgesEnabled()) return false;
    if (!isInstructionsJudge || !judgeInstructions) return false;
    const templateVariables = extractTemplateVariables(judgeInstructions);
    return templateVariables.includes('trace');
  }, [isInstructionsJudge, judgeInstructions]);

  // Check if the selected template is unsupported for running on sample traces.
  // Only applies when not all templates are supported (DB). Templates must either
  // be an instructions judge or have a chat-assessments mapping.
  const isUnsupportedTemplate = useMemo(() => {
    if (isRunningAllScorerTemplatesEnabled()) return false;
    if (isInstructionsJudge) return false;
    return !ASSESSMENT_NAME_TEMPLATE_MAPPING[llmTemplate as keyof typeof ASSESSMENT_NAME_TEMPLATE_MAPPING];
  }, [isInstructionsJudge, llmTemplate]);

  // Determine if run scorer button should be disabled
  const hasInstructionsError = Boolean((errors as any).instructions?.message);
  const isRetrievalRelevance = llmTemplate === LLM_TEMPLATE.RETRIEVAL_RELEVANCE;
  const hasEmptyGuidelines = isGuidelinesTemplate(llmTemplate) && (!guidelines || !guidelines.trim());

  // Determine tooltip message based on why the button is disabled
  const runScorerDisabledReason = useMemo(() => {
    // Highest precedence: template/scope not supported at all
    if (isUnsupportedTemplate) {
      return intl.formatMessage({
        defaultMessage: 'This judge template is not yet supported for sample judge output',
        description: 'Tooltip message when selected template is not supported for running on sample traces',
      });
    }

    if (!isEvaluatingSessionsInScorersEnabled() && isSessionLevelScorer) {
      return intl.formatMessage({
        defaultMessage: 'Running session level scorers is not yet supported',
        description: 'Tooltip message when scorer is session-level',
      });
    }

    // Model checks
    if (isScorerModelSelectionEnabled() && !modelValue) {
      return intl.formatMessage({
        defaultMessage: 'Please select a model to run the judge',
        description: 'Tooltip message when model is not selected',
      });
    }

    const modelProvider = getModelProvider(modelValue);
    const supportsDatabricks = shouldSupportRunningDatabricksProviderJudgesFromUI();
    const isUnsupportedModel =
      modelProvider === ModelProvider.OTHER || (modelProvider === ModelProvider.DATABRICKS && !supportsDatabricks);

    if (isUnsupportedModel) {
      const supportedProvider = supportsDatabricks ? 'databricks' : 'gateway';
      return intl.formatMessage(
        {
          defaultMessage:
            'Running the judge from the UI is only supported with {supportedProvider} endpoints, but the current model uses the {currentProvider} provider',
          description:
            'Tooltip message when model provider is not supported. supportedProvider is the required provider type, currentProvider is what the model currently uses.',
        },
        {
          supportedProvider,
          currentProvider: modelProvider,
        },
      );
    }

    // Judge-specific validation
    if (isInstructionsJudge) {
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
      if (isTraceVariableBlocked) {
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
    }

    if (selectedItemIds.length === 0) {
      return evaluationScope === ScorerEvaluationScope.TRACES
        ? intl.formatMessage({
            defaultMessage: 'Please select traces to run the judge',
            description: 'Tooltip message when no traces are selected',
          })
        : intl.formatMessage({
            defaultMessage: 'Please select sessions to run the judge',
            description: 'Tooltip message when no sessions are selected',
          });
    }
    return undefined;
  }, [
    isSessionLevelScorer,
    modelValue,
    isInstructionsJudge,
    judgeInstructions,
    hasInstructionsError,
    isTraceVariableBlocked,
    isRetrievalRelevance,
    isUnsupportedTemplate,
    hasEmptyGuidelines,
    selectedItemIds,
    evaluationScope,
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
      handleCancel={reset}
      handlePrevious={handlePrevious}
      handleNext={handleNext}
      totalTraces={data?.length ?? 0}
      selectedItemIds={selectedItemIds}
      onSelectedItemIdsChange={onSelectedItemIdsChange}
    />
  );
};

export default SampleScorerOutputPanelContainer;
