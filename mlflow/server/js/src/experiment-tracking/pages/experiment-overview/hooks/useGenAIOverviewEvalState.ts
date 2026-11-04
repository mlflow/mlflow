import { useEffect, useMemo } from 'react';
import {
  MLFLOW_RUN_TYPE_TAG,
  MLFLOW_RUN_TYPE_VALUE_EVALUATION,
  MLFLOW_RUN_TYPE_VALUE_GENAI_EVALUATE,
} from '../../../constants';
import { useExperimentEvaluationRunsData } from '../../../components/experiment-page/hooks/useExperimentEvaluationRunsData';
import type { RunEntity } from '../../../types';

const ACTIVE_RUN_STATUSES = new Set(['RUNNING', 'SCHEDULED']);
const getRunMetrics = (run: RunEntity) => run.data?.metrics ?? [];

export interface GenAIOverviewEvalScorePoint {
  runUuid: string;
  runName: string;
  datasetName?: string;
  timestampMs: number;
  scores: Record<string, number | null>;
}

export type GenAIOverviewEvalState =
  | { status: 'loading' }
  | { status: 'empty' }
  | {
      status: 'ready';
      runUuid: string;
      runCount: number;
      assessmentScoreNames: string[];
      scorePoints: GenAIOverviewEvalScorePoint[];
    }
  | { status: 'error' };

export const resolveGenAIOverviewEvalState = ({
  runs,
  isLoading,
  error,
  preferredAssessmentScoreNames = [],
}: {
  runs: readonly RunEntity[];
  isLoading: boolean;
  error: unknown;
  preferredAssessmentScoreNames?: string[];
}): GenAIOverviewEvalState => {
  if (isLoading) return { status: 'loading' };
  if (error) return { status: 'error' };
  if (runs.length === 0) return { status: 'empty' };

  const finishedRuns = runs.filter((run) => run.info.status === 'FINISHED');
  if (finishedRuns.length === 0) {
    return runs.some((run) => ACTIVE_RUN_STATUSES.has(run.info.status)) ? { status: 'loading' } : { status: 'empty' };
  }

  const availableAssessmentScoreNameSet = new Set<string>();
  const finiteMetricValuesByName = new Map<string, number[]>();
  for (const run of finishedRuns) {
    for (const metric of getRunMetrics(run)) {
      availableAssessmentScoreNameSet.add(metric.key);
      if (!Number.isFinite(metric.value)) continue;
      const values = finiteMetricValuesByName.get(metric.key) ?? [];
      values.push(metric.value);
      finiteMetricValuesByName.set(metric.key, values);
    }
  }
  const availableAssessmentScoreNames = Array.from(availableAssessmentScoreNameSet).sort((firstName, secondName) =>
    firstName.localeCompare(secondName),
  );
  const finiteAssessmentScoreNames = availableAssessmentScoreNames.filter((scoreName) =>
    finiteMetricValuesByName.has(scoreName),
  );
  const finiteAssessmentScoreNameSet = new Set(finiteAssessmentScoreNames);
  const boundedAssessmentScoreNames = finiteAssessmentScoreNames.filter((scoreName) =>
    finiteMetricValuesByName.get(scoreName)?.every((value) => value >= 0 && value <= 1),
  );
  const assessmentScoreNames = Array.from(
    new Set([
      ...preferredAssessmentScoreNames.filter((scoreName) => finiteAssessmentScoreNameSet.has(scoreName)),
      ...boundedAssessmentScoreNames,
      ...finiteAssessmentScoreNames,
      ...availableAssessmentScoreNames,
    ]),
  );
  const scorePoints = [...finishedRuns]
    .sort((firstRun, secondRun) => firstRun.info.startTime - secondRun.info.startTime)
    .map((run) => {
      const metricsByName = new Map(getRunMetrics(run).map((metric) => [metric.key, metric.value]));
      return {
        runUuid: run.info.runUuid,
        runName: run.info.runName,
        datasetName: run.inputs?.datasetInputs?.[0]?.dataset?.name,
        timestampMs: run.info.startTime,
        scores: Object.fromEntries(
          assessmentScoreNames.map((assessmentScoreName) => [
            assessmentScoreName,
            metricsByName.get(assessmentScoreName) ?? null,
          ]),
        ),
      };
    });

  const latestFinishedRun = finishedRuns.reduce((latestRun, run) =>
    run.info.startTime > latestRun.info.startTime ? run : latestRun,
  );

  return {
    status: 'ready',
    runUuid: latestFinishedRun.info.runUuid,
    runCount: finishedRuns.length,
    assessmentScoreNames,
    scorePoints,
  };
};

const useEvaluationRunsByType = (experimentId: string, runType: string) => {
  return useExperimentEvaluationRunsData({
    experimentId,
    enabled: Boolean(experimentId),
    filter: `tags.\`${MLFLOW_RUN_TYPE_TAG}\` = '${runType}'`,
    fetchAllPages: true,
  });
};

export const useGenAIOverviewEvalState = (experimentId: string): GenAIOverviewEvalState => {
  const legacyEvaluationRunsQuery = useEvaluationRunsByType(experimentId, MLFLOW_RUN_TYPE_VALUE_EVALUATION);
  const genAIEvaluationRunsQuery = useEvaluationRunsByType(experimentId, MLFLOW_RUN_TYPE_VALUE_GENAI_EVALUATE);
  const runs = useMemo(
    () =>
      Array.from(
        new Map(
          [...legacyEvaluationRunsQuery.data, ...genAIEvaluationRunsQuery.data].map((run) => [run.info.runUuid, run]),
        ).values(),
      ),
    [genAIEvaluationRunsQuery.data, legacyEvaluationRunsQuery.data],
  );
  const isLoading =
    legacyEvaluationRunsQuery.isLoading ||
    genAIEvaluationRunsQuery.isLoading ||
    Boolean(legacyEvaluationRunsQuery.hasNextPage) ||
    Boolean(genAIEvaluationRunsQuery.hasNextPage);
  const error = legacyEvaluationRunsQuery.error ?? genAIEvaluationRunsQuery.error;
  const hasActiveRun = runs.some((run) => ACTIVE_RUN_STATUSES.has(run.info.status));
  const refetchLegacyEvaluationRuns = legacyEvaluationRunsQuery.refetch;
  const refetchGenAIEvaluationRuns = genAIEvaluationRunsQuery.refetch;

  useEffect(() => {
    if (isLoading || error || !hasActiveRun) return undefined;
    const interval = window.setInterval(() => {
      void Promise.all([refetchLegacyEvaluationRuns(), refetchGenAIEvaluationRuns()]);
    }, 5000);
    return () => window.clearInterval(interval);
  }, [error, hasActiveRun, isLoading, refetchGenAIEvaluationRuns, refetchLegacyEvaluationRuns]);

  return useMemo(() => resolveGenAIOverviewEvalState({ runs, isLoading, error }), [error, isLoading, runs]);
};
