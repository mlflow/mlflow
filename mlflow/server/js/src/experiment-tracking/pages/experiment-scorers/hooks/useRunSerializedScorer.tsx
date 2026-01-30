import { useCallback } from 'react';
import { useEvaluateTraces } from '../useEvaluateTraces';
import { EvaluateTracesParams, LLM_TEMPLATE, LLMScorer, ScheduledScorer } from '../types';
import { ScorerEvaluationScope } from '../constants';
import { transformScheduledScorer } from '../utils/scorerTransformUtils';
import { ScorerFinishedEvent } from '../useEvaluateTracesAsync';
import { isObject } from 'lodash';
import { useTemplateOptions } from '../llmScorerUtils';
import { TEMPLATE_INSTRUCTIONS_MAP } from '../prompts';

/**
 * Runs a known serialized scorer on a set of traces.
 */
export const useRunSerializedScorer = ({
  experimentId,
  onScorerFinished,
  scope = ScorerEvaluationScope.TRACES,
}: {
  experimentId?: string;
  onScorerFinished?: (event: ScorerFinishedEvent) => void;
  scope?: ScorerEvaluationScope;
}) => {
  const [evaluateTracesFn, { latestEvaluation, isLoading, allEvaluations }] = useEvaluateTraces({ onScorerFinished });
  const { displayMap } = useTemplateOptions(scope);

  const getEvaluationParams = useCallback(
    (scorerOrTemplate: LLMScorer | LLM_TEMPLATE, traceIds: string[], endpointName?: string): EvaluateTracesParams => {
      if (!experimentId) {
        throw new Error('Experiment ID is required');
      }

      const baseParams: Omit<EvaluateTracesParams, 'judgeInstructions' | 'serializedScorer'> = {
        itemCount: undefined,
        itemIds: traceIds,
        locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' as const }],
        experimentId,
        evaluationScope: scope,
        saveAssessment: true,
      };
      if (isObject(scorerOrTemplate)) {
        const scorer = scorerOrTemplate;
        const scorerConfig = transformScheduledScorer(scorer);
        return {
          ...baseParams,
          judgeInstructions: scorer.instructions || '',
          serializedScorer: scorerConfig.serialized_scorer,
        };
      }

      const instructionsForCustomTemplate = TEMPLATE_INSTRUCTIONS_MAP[scorerOrTemplate];

      // Built an ad-hoc custom LLM-as-a-judge scorer.
      // If the template has instructions, use them as 'Custom' judge. Otherwise, assume it's a pre-built template.
      const adHocScheduledScorer: ScheduledScorer = instructionsForCustomTemplate
        ? {
            name: displayMap[scorerOrTemplate],
            type: 'llm',
            llmTemplate: 'Custom',
            instructions: instructionsForCustomTemplate,
            model: endpointName,
            is_instructions_judge: true,
            isSessionLevelScorer: scope === ScorerEvaluationScope.SESSIONS,
          }
        : {
            name: displayMap[scorerOrTemplate],
            type: 'llm',
            llmTemplate: scorerOrTemplate,
            model: endpointName,
            is_instructions_judge: false,
            isSessionLevelScorer: scope === ScorerEvaluationScope.SESSIONS,
          };

      // Create backend-compatible scorer config
      const scorerConfig = transformScheduledScorer(adHocScheduledScorer);

      return {
        ...baseParams,
        judgeInstructions: TEMPLATE_INSTRUCTIONS_MAP[scorerOrTemplate as LLM_TEMPLATE],
        serializedScorer: scorerConfig.serialized_scorer,
      };
    },
    [displayMap, experimentId, scope],
  );

  const evaluateTraces = useCallback(
    (scorerOrTemplate: LLMScorer | LLM_TEMPLATE, traceIds: string[], endpointName?: string) => {
      const scorer = scorerOrTemplate;
      const evaluationParams = getEvaluationParams(scorer, traceIds, endpointName);
      return evaluateTracesFn(evaluationParams);
    },
    [getEvaluationParams, evaluateTracesFn],
  );

  return {
    evaluateTraces,
    latestEvaluation,
    isLoading,
    allEvaluations,
  };
};
