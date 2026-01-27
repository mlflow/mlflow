import { useCallback } from 'react';
import { useEvaluateTraces } from '../useEvaluateTraces';
import { EvaluateTracesParams, LLM_TEMPLATE, LLMScorer, ScheduledScorer } from '../types';
import { ASSESSMENT_NAME_TEMPLATE_MAPPING, ScorerEvaluationScope } from '../constants';
import { transformScheduledScorer } from '../utils/scorerTransformUtils';

/**
 * Runs a known serialized scorer on a set of traces.
 */
export const useRunSerializedScorer = ({
  experimentId,
  onScorerFinished,
}: {
  experimentId?: string;
  onScorerFinished?: () => void;
}) => {
  const [evaluateTracesFn, { data, isLoading }] = useEvaluateTraces({ onScorerFinished });

  const evaluateTraces = useCallback(
    (scorer: LLMScorer, traceIds: string[]) => {
      if (!experimentId) {
        throw new Error('Experiment ID is required');
      }
      const scorerConfig = transformScheduledScorer(scorer);
      // Prepare evaluation parameters based on mode
      const evaluationParams: EvaluateTracesParams = {
        itemCount: undefined,
        itemIds: traceIds,
        locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' as const }],
        judgeInstructions: scorer.instructions || '',
        experimentId,
        evaluationScope: ScorerEvaluationScope.TRACES,
        serializedScorer: scorerConfig.serialized_scorer,
      };
      return evaluateTracesFn(evaluationParams);
    },
    [evaluateTracesFn, experimentId],
  );

  return {
    evaluateTraces,
    data,
    isLoading,
  };
};
