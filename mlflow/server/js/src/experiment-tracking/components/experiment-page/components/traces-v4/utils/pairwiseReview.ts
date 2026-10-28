import {
  getAssessmentValue,
  type Assessment,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';

export const PAIRWISE_PREFERENCE_ASSESSMENT_NAME = 'Pairwise preference';
export const PAIRWISE_SCORE_WIN_VALUE = 1;
export const PAIRWISE_SCORE_LOSE_VALUE = 0;
export const PAIRWISE_SCORE_OPPONENT_TRACE_ID_METADATA_KEY = 'mlflow.assessment.pairwiseOpponentTraceId';

export type PairwiseScoreValue = typeof PAIRWISE_SCORE_WIN_VALUE | typeof PAIRWISE_SCORE_LOSE_VALUE;

export const getPairwiseScoreValue = (assessment: Assessment): PairwiseScoreValue | undefined => {
  if (!('feedback' in assessment)) {
    return undefined;
  }
  const value = getAssessmentValue(assessment);
  return value === PAIRWISE_SCORE_WIN_VALUE || value === PAIRWISE_SCORE_LOSE_VALUE ? value : undefined;
};

export const getActivePairwiseScoreAssessments = (trace: ModelTraceInfoV3): Assessment[] => {
  const pairwiseAssessments = (trace.assessments ?? []).filter(
    (assessment) =>
      assessment.valid !== false &&
      assessment.assessment_name === PAIRWISE_PREFERENCE_ASSESSMENT_NAME &&
      getPairwiseScoreValue(assessment) !== undefined,
  );
  const overriddenAssessmentIds = new Set(
    pairwiseAssessments.flatMap((assessment) => (assessment.overrides ? [assessment.overrides] : [])),
  );
  const activeAssessments = pairwiseAssessments.filter(
    (assessment) => !overriddenAssessmentIds.has(assessment.assessment_id),
  );
  const latestAssessmentByPair = new Map<string, Assessment>();
  for (const assessment of activeAssessments) {
    const opponentTraceId = assessment.metadata?.[PAIRWISE_SCORE_OPPONENT_TRACE_ID_METADATA_KEY];
    const pairKey = opponentTraceId
      ? `${assessment.source.source_type}:${assessment.source.source_id}:${opponentTraceId}`
      : assessment.assessment_id;
    const previous = latestAssessmentByPair.get(pairKey);
    const assessmentTime = assessment.last_update_time ?? assessment.create_time ?? '';
    const previousTime = previous?.last_update_time ?? previous?.create_time ?? '';
    if (!previous || assessmentTime >= previousTime) {
      latestAssessmentByPair.set(pairKey, assessment);
    }
  }
  return [...latestAssessmentByPair.values()];
};

export interface PairwiseScoreCounts {
  wins: number;
  losses: number;
}

export const getPairwiseScoreCounts = (
  trace: ModelTraceInfoV3,
  opponentTraceIds?: ReadonlySet<string>,
): PairwiseScoreCounts =>
  getActivePairwiseScoreAssessments(trace).reduce<PairwiseScoreCounts>(
    (counts, assessment) => {
      const opponentTraceId = assessment.metadata?.[PAIRWISE_SCORE_OPPONENT_TRACE_ID_METADATA_KEY];
      if (opponentTraceIds && (!opponentTraceId || !opponentTraceIds.has(opponentTraceId))) {
        return counts;
      }
      const value = getPairwiseScoreValue(assessment);
      if (value === PAIRWISE_SCORE_WIN_VALUE) {
        counts.wins += 1;
      } else if (value === PAIRWISE_SCORE_LOSE_VALUE) {
        counts.losses += 1;
      }
      return counts;
    },
    { wins: 0, losses: 0 },
  );

export const findPairwiseScoreAssessment = ({
  trace,
  opponentTraceId,
  sourceId,
}: {
  trace: ModelTraceInfoV3;
  opponentTraceId: string;
  sourceId: string;
}): Assessment | undefined =>
  getActivePairwiseScoreAssessments(trace)
    .filter(
      (assessment) =>
        assessment.source.source_id === sourceId &&
        assessment.metadata?.[PAIRWISE_SCORE_OPPONENT_TRACE_ID_METADATA_KEY] === opponentTraceId,
    )
    .reduce<Assessment | undefined>((latest, assessment) => {
      if (!latest) {
        return assessment;
      }
      return (assessment.create_time ?? '') > (latest.create_time ?? '') ? assessment : latest;
    }, undefined);
