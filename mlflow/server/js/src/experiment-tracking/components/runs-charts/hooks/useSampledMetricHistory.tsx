import { chunk } from 'lodash';
import { useCallback, useMemo } from 'react';
import type { SampledMetricsByRunUuidState, MetricEntity } from '@mlflow/mlflow/src/experiment-tracking/types';
import { EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL } from '../../../utils/MetricsUtils';
import { shouldEnableGraphQLSampledMetrics } from '../../../../common/utils/FeatureUtils';
import { useSampledMetricHistoryGraphQL } from './useSampledMetricHistoryGraphQL';
import { useQueries, type QueryFunctionContext } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { fetchOrFail, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import { stringify as queryStringStringify } from 'qs';

type SampledMetricData = SampledMetricsByRunUuidState[string][string][string];

export type SampledMetricsByRun = {
  runUuid: string;
} & {
  [metricKey: string]: SampledMetricData;
};

const SAMPLED_METRIC_HISTORY_API_RUN_LIMIT = 100;

interface GetHistoryBulkIntervalResponseType {
  metrics: (MetricEntity & { run_id: string })[];
}

type SampledMetricHistoryQueryKey = [
  'sampledMetricHistory',
  {
    runUuids: string[];
    metricKey: string;
    maxResults?: number;
    range?: [number, number];
  },
];

/**
 * Automatically fetches sampled metric history for runs, used in run runs charts.
 * React Query-based implementation that leverages built-in caching and refresh capabilities.
 * Also backfills Redux store to maintain compatibility with existing code.
 */
const useSampledMetricHistoryREST = (params: {
  runUuids: string[];
  metricKeys: string[];
  maxResults?: number;
  range?: [number, number];
  enabled?: boolean;
  autoRefreshEnabled?: boolean;
}) => {
  const { metricKeys, runUuids, enabled = true, maxResults, range, autoRefreshEnabled = false } = params;

  // Create query function for fetching metric history and backfilling Redux
  const queryFn = useCallback(async ({ queryKey, signal }: QueryFunctionContext<SampledMetricHistoryQueryKey>) => {
    const [, { runUuids, metricKey, maxResults, range }] = queryKey;

    const queryParamsInput: {
      run_ids: string[];
      metric_key: string;
      max_results?: string;
      start_step?: string;
      end_step?: string;
    } = {
      run_ids: runUuids,
      metric_key: decodeURIComponent(metricKey),
    };

    if (maxResults !== undefined) {
      queryParamsInput.max_results = maxResults.toString();
    }

    if (range) {
      const [start_step, end_step] = range;
      queryParamsInput.start_step = start_step.toString();
      queryParamsInput.end_step = end_step.toString();
    }

    const queryParams = queryStringStringify(queryParamsInput, { arrayFormat: 'repeat' });

    const response = await fetchOrFail(
      getAjaxUrl(`ajax-api/2.0/mlflow/metrics/get-history-bulk-interval?${queryParams}`),
      { signal },
    );

    return response.json() as Promise<GetHistoryBulkIntervalResponseType>;
  }, []);

  // Create queries for all combinations of metric keys and chunked run UUIDs
  const queries = useMemo(() => {
    const allQueries: Array<{
      queryKey: SampledMetricHistoryQueryKey;
      queryFn: typeof queryFn;
      enabled: boolean;
      refetchInterval: number | false;
      staleTime: number;
    }> = [];

    metricKeys.forEach((metricKey) => {
      const runUuidChunks = chunk(runUuids, SAMPLED_METRIC_HISTORY_API_RUN_LIMIT);
      runUuidChunks.forEach((runUuidsChunk) => {
        allQueries.push({
          queryKey: ['sampledMetricHistory', { runUuids: runUuidsChunk, metricKey, maxResults, range }],
          queryFn,
          enabled,
          refetchInterval: autoRefreshEnabled ? EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL : false,
          staleTime: autoRefreshEnabled ? EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL : Infinity,
        });
      });
    });

    return allQueries;
  }, [metricKeys, runUuids, maxResults, range, enabled, autoRefreshEnabled, queryFn]);

  const queryResults = useQueries({ queries });

  // Transform query results into the expected format
  const { resultsByRunUuid, isLoading, isRefreshing } = useMemo(() => {
    let anyLoading = false;
    let anyRefreshing = false;

    const metricDataByRunUuid: Record<string, SampledMetricsByRun> = {};

    queryResults.forEach((queryResult) => {
      anyLoading = anyLoading || queryResult.isLoading;
      anyRefreshing = anyRefreshing || queryResult.isFetching;

      if (queryResult.data?.metrics && queryResult.data.metrics.length > 0) {
        const metricKey = queryResult.data.metrics[0].key;

        queryResult.data.metrics.forEach((metric) => {
          const runUuid = metric.run_id;

          if (!metricDataByRunUuid[runUuid]) {
            metricDataByRunUuid[runUuid] = { runUuid } as SampledMetricsByRun;
          }

          if (!metricDataByRunUuid[runUuid][metricKey]) {
            metricDataByRunUuid[runUuid][metricKey] = {
              loading: queryResult.isLoading,
              refreshing: queryResult.isFetching,
              metricsHistory: [],
              lastUpdatedTime: Date.now(),
            };
          }

          const metricsHistory = metricDataByRunUuid[runUuid][metricKey].metricsHistory;
          if (metricsHistory) {
            metricsHistory.push(metric);
          }
        });
      }
    });

    return {
      resultsByRunUuid: metricDataByRunUuid,
      isLoading: anyLoading,
      isRefreshing: anyRefreshing,
    };
  }, [queryResults]);

  // Manual refresh function
  const refresh = useCallback(() => {
    queryResults.forEach((queryResult) => {
      queryResult.refetch();
    });
  }, [queryResults]);

  return { isLoading, isRefreshing, resultsByRunUuid, refresh };
};

/**
 * A switcher hook that selects between the REST and GraphQL implementations of the
 * `useSampledMetricHistory` hook based on flags and parameter context.
 */
export const useSampledMetricHistory = (params: {
  runUuids: string[];
  metricKeys: string[];
  maxResults?: number;
  range?: [number, number];
  enabled?: boolean;
  autoRefreshEnabled?: boolean;
}) => {
  const { metricKeys, enabled, autoRefreshEnabled, runUuids } = params;

  // We should use the apollo hook if there is only one metric key and the number of runUuids is less than 100.
  // To be improved after endpoint will start supporting multiple metric keys.
  const shouldUseGraphql = shouldEnableGraphQLSampledMetrics() && metricKeys.length === 1 && runUuids.length <= 100;

  const legacyResult = useSampledMetricHistoryREST({
    ...params,
    enabled: enabled && !shouldUseGraphql,
    autoRefreshEnabled: autoRefreshEnabled && !shouldUseGraphql,
  });

  const graphQlResult = useSampledMetricHistoryGraphQL({
    ...params,
    metricKey: metricKeys[0],
    enabled: enabled && shouldUseGraphql,
    autoRefreshEnabled: autoRefreshEnabled && shouldUseGraphql,
  });

  return shouldUseGraphql ? graphQlResult : legacyResult;
};
