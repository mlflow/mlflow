import { useMemo } from 'react';
import { useIntl } from 'react-intl';
import { useGenAiExperimentRunsForComparison } from '@databricks/web-shared/genai-traces-table';
import {
  buildEvalRunComparisonSnapshot,
  normalizeComparisonRunUuids,
  type EvalRunComparisonSnapshot,
} from './evalRunComparisonData';
import { aggregateEvalRunConfigurationTraces, type EvalRunConfigurationTraceStats } from './evalRunConfigurationData';
import { buildEvalRunComparisonScorers, type EvalRunComparisonScorerRun } from './evalRunComparisonScorerData';
import { useEvalRunComparisonRunTraces } from './useEvalRunComparisonRunTraces';

export interface EvalRunConfigurationTraceData {
  runUuid: string;
  runName: string;
  stats?: EvalRunConfigurationTraceStats;
  status: 'loading' | 'unavailable' | 'error' | 'ready';
}

export interface UseEvalRunComparisonDataArgs {
  experimentId: string;
  baselineRunUuid: string;
  compareRunUuids: string[];
}

export interface EvalRunComparisonData {
  snapshot: EvalRunComparisonSnapshot | undefined;
  configurationRuns: EvalRunConfigurationTraceData[];
  scorerRuns: EvalRunComparisonScorerRun[];
  noComparison: boolean;
  isLoading: boolean;
  failedRunUuids: string[];
  refetch: () => Promise<void>;
}

export const useEvalRunComparisonData = ({
  experimentId,
  baselineRunUuid,
  compareRunUuids,
}: UseEvalRunComparisonDataArgs): EvalRunComparisonData => {
  const intl = useIntl();
  const runUuids = useMemo(
    () => normalizeComparisonRunUuids(baselineRunUuid, compareRunUuids),
    [baselineRunUuid, compareRunUuids],
  );
  const noComparison = runUuids.length < 2;
  const { runInfos, isLoading: isLoadingRunNames } = useGenAiExperimentRunsForComparison(experimentId, noComparison);
  const {
    runsWithTraces,
    isLoading,
    isFetching,
    isDataReady,
    failedRunUuids,
    readyRunUuids,
    fetchedTraceCounts,
    traceLimit,
    refetch,
  } = useEvalRunComparisonRunTraces({ experimentId, runUuids, disabled: noComparison });
  const configurationRuns = useMemo(
    () =>
      runsWithTraces.map<EvalRunConfigurationTraceData>(({ runUuid, traceInfos }, index) => {
        const status = noComparison
          ? 'unavailable'
          : failedRunUuids.includes(runUuid)
            ? 'error'
            : readyRunUuids.includes(runUuid)
              ? 'ready'
              : 'loading';
        return {
          runUuid,
          runName: runInfos?.find((run) => run.runUuid === runUuid)?.runName || runUuid,
          status,
          stats:
            status === 'ready'
              ? aggregateEvalRunConfigurationTraces(traceInfos, fetchedTraceCounts[index] >= traceLimit)
              : undefined,
        };
      }),
    [runsWithTraces, runInfos, noComparison, failedRunUuids, readyRunUuids, fetchedTraceCounts, traceLimit],
  );
  const scorerRuns = useMemo(
    () =>
      configurationRuns.map<EvalRunComparisonScorerRun>((run, index) => ({
        runUuid: run.runUuid,
        runName: run.runName,
        status: run.status,
        isTruncated: fetchedTraceCounts[index] >= traceLimit,
        scorers: run.status === 'ready' ? buildEvalRunComparisonScorers(intl, runsWithTraces[index].traceInfos) : [],
      })),
    [configurationRuns, fetchedTraceCounts, traceLimit, intl, runsWithTraces],
  );
  const snapshot = useMemo(() => {
    if (!isDataReady || isFetching || isLoadingRunNames) {
      return undefined;
    }
    const names = new Map(runInfos?.flatMap((run) => (run.runUuid ? [[run.runUuid, run.runName || run.runUuid]] : [])));
    const truncated = new Set(runUuids.filter((_, index) => fetchedTraceCounts[index] >= traceLimit));
    return buildEvalRunComparisonSnapshot(intl, runsWithTraces, names, truncated);
  }, [
    intl,
    isDataReady,
    isFetching,
    isLoadingRunNames,
    runInfos,
    runUuids,
    fetchedTraceCounts,
    traceLimit,
    runsWithTraces,
  ]);
  return {
    snapshot,
    configurationRuns,
    scorerRuns,
    noComparison,
    isLoading:
      isLoading || isFetching || isLoadingRunNames || (!isDataReady && !failedRunUuids.length && !noComparison),
    failedRunUuids,
    refetch,
  };
};
