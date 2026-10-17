import { useMemo } from 'react';
import { useIntl } from 'react-intl';
import { useQuery } from '@databricks/web-shared/query-client';
import {
  convertTraceInfoV3ToRunEvalEntry,
  createTraceLocationForExperiment,
  getAssessmentInfos,
  getEvalTabTotalTracesLimit,
  SEARCH_MLFLOW_TRACES_QUERY_KEY,
  searchMlflowTracesQueryFn,
} from '@databricks/web-shared/genai-traces-table';
// Not re-exported from the genai-traces-table barrel in OSS — import from its module.
import { filterTracesByAssessmentSourceRunId } from '@databricks/web-shared/genai-traces-table/utils/TraceUtils';
import { useGetRunQuery } from '../../components/run-page/hooks/useGetRunQuery';
import { aggregateEvalRunSidebarTraceStats } from './evalRunConfigurationData';
import { buildEvalRunTraceFilter } from './useEvalRunTracesV4Controller';
import { getConsistentScorerReference } from './scorerLineage';
import { parseLinkedPromptsFromRunTags } from '../prompts/utils';
import type { RunDetailsScorer, RunDetailsSidebarProps, RunDetailsStats } from './RunDetailsSidebar';

const ENDPOINT_KEYS = ['endpoint', 'model', 'model_name', 'served_entity'];

export interface UseEvalRunSidebarDataArgs {
  experimentId: string;
  runUuid: string;
}

export interface EvalRunSidebarData {
  sidebarProps: RunDetailsSidebarProps;
  isLoading: boolean;
  error: unknown;
  isTruncated: boolean;
  hasPartialUsage: boolean;
}

const findValue = <V,>(
  entries: { key: string | null; value: V | null }[] | null | undefined,
  candidates: string[],
): V | undefined => entries?.find((entry) => entry.key !== null && candidates.includes(entry.key))?.value ?? undefined;

export const useEvalRunSidebarData = ({ experimentId, runUuid }: UseEvalRunSidebarDataArgs): EvalRunSidebarData => {
  const intl = useIntl();
  const { data: run, loading: runLoading, apolloError, apiError } = useGetRunQuery({ runUuid });
  const traceLimit = getEvalTabTotalTracesLimit();
  const tracesQuery = useQuery({
    queryKey: [SEARCH_MLFLOW_TRACES_QUERY_KEY, 'eval-run-sidebar', experimentId, runUuid, traceLimit],
    queryFn: ({ signal }) =>
      searchMlflowTracesQueryFn({
        signal,
        locations: [createTraceLocationForExperiment(experimentId)],
        filter: buildEvalRunTraceFilter({ runUuid }),
        limit: traceLimit,
      }),
    refetchOnWindowFocus: false,
  });
  const fetchedTraces = tracesQuery.data;
  const traceInfos = useMemo(
    () => filterTracesByAssessmentSourceRunId(fetchedTraces, runUuid) ?? [],
    [fetchedTraces, runUuid],
  );
  const traceStats = useMemo(
    () =>
      fetchedTraces ? aggregateEvalRunSidebarTraceStats(traceInfos, fetchedTraces.length >= traceLimit) : undefined,
    [fetchedTraces, traceInfos, traceLimit],
  );
  const scorers = useMemo<RunDetailsScorer[]>(
    () =>
      fetchedTraces
        ? getAssessmentInfos(
            intl,
            traceInfos.map((trace) => convertTraceInfoV3ToRunEvalEntry(trace)),
            [],
          ).map((assessmentInfo) => ({
            name: assessmentInfo.name,
            displayName: assessmentInfo.displayName,
            scorerReference: getConsistentScorerReference(traceInfos, assessmentInfo.name),
          }))
        : [],
    [fetchedTraces, intl, traceInfos],
  );

  const runDetails = useMemo(() => {
    const dataset = run?.inputs?.datasetInputs?.[0]?.dataset;
    const rawStartTime = run?.info?.startTime;
    const parsedStartTime =
      rawStartTime !== null && rawStartTime !== undefined && rawStartTime !== '' ? Number(rawStartTime) : undefined;
    const runTags = (run?.data?.tags ?? []).flatMap((tag) =>
      tag.key === null ? [] : [{ key: tag.key, value: tag.value ?? '' }],
    );
    const tags = runTags.filter((tag) => !tag.key.startsWith('mlflow.'));
    const prompts = parseLinkedPromptsFromRunTags(Object.fromEntries(runTags.map((tag) => [tag.key, tag])));

    return {
      experimentId,
      createdBy: run?.info?.userId ?? undefined,
      createdAt: parsedStartTime !== undefined && Number.isFinite(parsedStartTime) ? parsedStartTime : undefined,
      dataset: dataset ?? undefined,
      prompts,
      endpoint: findValue(run?.data?.params, ENDPOINT_KEYS) ?? findValue(run?.data?.tags, ENDPOINT_KEYS) ?? undefined,
      tags,
    };
  }, [experimentId, run]);

  const sidebarStats = useMemo<RunDetailsStats>(
    () =>
      traceStats
        ? {
            traces: traceStats.traceCount,
            tracesTruncated: traceStats.isTruncated,
            totalTokens: traceStats.totalTokens,
            totalCost: traceStats.totalCost,
            errors: traceStats.errors,
          }
        : {},
    [traceStats],
  );
  const sidebarProps = useMemo<RunDetailsSidebarProps>(
    () => ({
      ...runDetails,
      stats: sidebarStats,
      scorers,
    }),
    [runDetails, sidebarStats, scorers],
  );

  return {
    sidebarProps,
    isLoading: runLoading || tracesQuery.isLoading,
    error: apolloError ?? apiError ?? tracesQuery.error ?? undefined,
    isTruncated: traceStats?.isTruncated ?? false,
    hasPartialUsage: traceStats?.hasPartialUsage ?? false,
  };
};
