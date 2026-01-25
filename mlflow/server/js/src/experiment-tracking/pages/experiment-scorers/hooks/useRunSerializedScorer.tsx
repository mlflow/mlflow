import { useCallback } from 'react';
import { useEvaluateTraces } from '../useEvaluateTraces';
import { ScorerUpdateEvent } from '../useEvaluateTracesAsync';
import { EvaluateTracesParams, LLMScorer } from '../types';
import { ASSESSMENT_NAME_TEMPLATE_MAPPING, ScorerEvaluationScope } from '../constants';
import { transformScheduledScorer } from '../utils/scorerTransformUtils';

/**
 * Generate a unique request key for a scorer evaluation
 */
const generateRequestKey = (scorerName: string, traceIds: string[]) => {
  // Create a deterministic key based on scorer name and trace IDs
  return `${scorerName}-${traceIds.sort().join('-')}`;
};

/**
 * Runs a known serialized scorer on a set of traces.
 * Supports multiple concurrent evaluations keyed by scorer name and trace IDs.
 */
export const useRunSerializedScorer = ({
  experimentId,
  onScorerUpdate,
}: {
  experimentId?: string;
  /** Callback fired when an evaluation's status changes */
  onScorerUpdate?: (event: ScorerUpdateEvent) => void;
}) => {
  const [evaluateTracesFn, { data, isLoading, getEvaluation, allEvaluations }] = useEvaluateTraces({ onScorerUpdate });

  const evaluateTraces = useCallback(
    async (scorer: LLMScorer, traceIds: string[]) => {
      if (!experimentId) {
        throw new Error('Experiment ID is required');
      }
      try {
        const scorerConfig = transformScheduledScorer(scorer);
        // Generate a unique request key for this evaluation
        const requestKey = generateRequestKey(scorer.name, traceIds);

        // Prepare evaluation parameters based on mode
        const evaluationParams: EvaluateTracesParams = {
          itemCount: undefined,
          itemIds: traceIds,
          locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' as const }],
          requestedAssessments: [
            {
              assessment_name:
                ASSESSMENT_NAME_TEMPLATE_MAPPING[scorer.llmTemplate as keyof typeof ASSESSMENT_NAME_TEMPLATE_MAPPING],
            },
          ],
          judgeInstructions: scorer.instructions || '',
          experimentId,
          evaluationScope: ScorerEvaluationScope.TRACES,
          serializedScorer: scorerConfig.serialized_scorer,
          saveAssessment: true,
        };
        return await evaluateTracesFn(evaluationParams, requestKey);
      } catch (error) {
        // TODO: Handle serialization error
        return undefined;
      }
    },
    [evaluateTracesFn, experimentId],
  );

  return {
    evaluateTraces,
    data,
    isLoading,
    // Extended API for accessing per-request state
    getEvaluation,
    allEvaluations,
  };
};
