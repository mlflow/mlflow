import { useCallback, useMemo } from 'react';
import { useQueries } from '@databricks/web-shared/query-client';
import {
  createTraceLocationForExperiment,
  getEvalTabTotalTracesLimit,
  SEARCH_MLFLOW_TRACES_QUERY_KEY,
  searchMlflowTracesQueryFn,
} from '@databricks/web-shared/genai-traces-table';
// Not re-exported from the genai-traces-table barrel in OSS — import from its module.
import { filterTracesByAssessmentSourceRunId } from '@databricks/web-shared/genai-traces-table/utils/TraceUtils';
import type { EvalComparisonRunWithTraces } from './evalRunComparisonData';
import { buildEvalRunTraceFilter } from './useEvalRunTracesV4Controller';

/** Searches each compared run's traces on the experiment, one query per run, as on the single-run page. */
export const useEvalRunComparisonRunTraces = ({
  experimentId,
  runUuids,
  disabled,
}: {
  experimentId: string;
  runUuids: string[];
  disabled?: boolean;
}) => {
  const traceLimit = getEvalTabTotalTracesLimit();
  // Disabled lookups create no query observers.
  const activeRunUuids = useMemo(() => (disabled ? [] : runUuids), [disabled, runUuids]);
  const queryResults = useQueries({
    queries: activeRunUuids.map((runUuid) => ({
      queryKey: [SEARCH_MLFLOW_TRACES_QUERY_KEY, 'eval-run-comparison', experimentId, runUuid, traceLimit],
      queryFn: ({ signal }: { signal?: AbortSignal }) =>
        searchMlflowTracesQueryFn({
          signal,
          locations: [createTraceLocationForExperiment(experimentId)],
          filter: buildEvalRunTraceFilter({ runUuid }),
          limit: traceLimit,
        }),
      refetchOnWindowFocus: false,
    })),
  });

  const isLoading = !disabled && queryResults.some((result) => result.isLoading);
  const isFetching = !disabled && queryResults.some((result) => result.isFetching);
  const failedRunUuids = useMemo(
    () => activeRunUuids.filter((_, index) => queryResults[index]?.isError),
    [activeRunUuids, queryResults],
  );
  // Cached results stay usable during a refetch, including successful searches with no traces.
  const readyRunUuids = useMemo(
    () =>
      activeRunUuids.filter((_, index) => queryResults[index]?.isSuccess && queryResults[index]?.data !== undefined),
    [activeRunUuids, queryResults],
  );
  const isDataReady = !disabled && queryResults.length === runUuids.length && readyRunUuids.length === runUuids.length;
  const fetchedTraceCounts = useMemo(() => queryResults.map((result) => result.data?.length ?? 0), [queryResults]);
  const refetch = useCallback(async () => {
    await Promise.all(queryResults.map((result) => result.refetch({ cancelRefetch: false })));
  }, [queryResults]);
  const runsWithTraces = useMemo<EvalComparisonRunWithTraces[]>(
    () =>
      runUuids.map((runUuid, index) => ({
        runUuid,
        // A trace linked to several runs keeps only the assessments this run produced.
        traceInfos: filterTracesByAssessmentSourceRunId(queryResults[index]?.data ?? [], runUuid) ?? [],
      })),
    [runUuids, queryResults],
  );

  return {
    runsWithTraces,
    isLoading,
    isFetching,
    isDataReady,
    failedRunUuids,
    readyRunUuids,
    fetchedTraceCounts,
    traceLimit,
    refetch,
  };
};
