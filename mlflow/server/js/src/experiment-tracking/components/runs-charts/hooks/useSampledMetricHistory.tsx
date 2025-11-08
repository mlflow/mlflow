import { chunk, isEqual, keyBy } from 'lodash';
import { useCallback, useEffect, useMemo, useRef } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import type { ReduxState, ThunkDispatch } from '../../../../redux-types';
import { createChartAxisRangeKey } from '../components/RunsCharts.common';
import { getSampledMetricHistoryBulkAction } from '../../../sdk/SampledMetricHistoryService';
import type { SampledMetricsByRunUuidState } from '@mlflow/mlflow/src/experiment-tracking/types';
import { EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL } from '../../../utils/MetricsUtils';
import Utils from '../../../../common/utils/Utils';
import { shouldEnableGraphQLSampledMetrics } from '../../../../common/utils/FeatureUtils';
import { useSampledMetricHistoryGraphQL } from './useSampledMetricHistoryGraphQL';

type SampledMetricData = SampledMetricsByRunUuidState[string][string][string];

export type SampledMetricsByRun = {
  runUuid: string;
} & {
  [metricKey: string]: SampledMetricData;
};

const SAMPLED_METRIC_HISTORY_API_RUN_LIMIT = 100;

/**
 * Automatically fetches sampled metric history for runs, used in run runs charts.
 * After updating list of metrics or runs, optimizes the request and fetches
 * only the missing entries.
 *
 * REST-based implementation.
 */
const useSampledMetricHistoryREST = (params: {
  runUuids: string[];
  metricKeys: string[];
  maxResults?: number;
  range?: [number, number];
  enabled?: boolean;
  autoRefreshEnabled?: boolean;
}) => {
  const { metricKeys, runUuids, enabled, maxResults, range, autoRefreshEnabled } = params;
  const dispatch = useDispatch<ThunkDispatch>();

  const { resultsByRunUuid, isLoading, isRefreshing } = useSelector(
    (store: ReduxState) => {
      const rangeKey = createChartAxisRangeKey(range);

      let anyRunRefreshing = false;
      let anyRunLoading = false;

      const returnValues: SampledMetricsByRun[] = runUuids.map((runUuid) => {
        const metricsByMetricKey = metricKeys.reduce(
          (dataByMetricKey: { [key: string]: SampledMetricData }, metricKey: string) => {
            const runMetricData = store.entities.sampledMetricsByRunUuid[runUuid]?.[metricKey]?.[rangeKey];

            if (!runMetricData) {
              return dataByMetricKey;
            }

            anyRunLoading = anyRunLoading || Boolean(runMetricData.loading);
            anyRunRefreshing = anyRunRefreshing || Boolean(runMetricData.refreshing);

            dataByMetricKey[metricKey] = runMetricData;
            return dataByMetricKey;
          },
          {},
        );

        return {
          runUuid,
          ...metricsByMetricKey,
        };
      });

      return {
        isLoading: anyRunLoading,
        isRefreshing: anyRunRefreshing,
        resultsByRunUuid: keyBy(returnValues, 'runUuid'),
      };
    },
    (left, right) =>
      isEqual(left.resultsByRunUuid, right.resultsByRunUuid) &&
      left.isLoading === right.isLoading &&
      left.isRefreshing === right.isRefreshing,
  );

  const refreshFn = useCallback(() => {
    metricKeys.forEach((metricKey) => {
      chunk(runUuids, SAMPLED_METRIC_HISTORY_API_RUN_LIMIT).forEach((runUuidsChunk) => {
        const action = getSampledMetricHistoryBulkAction(runUuidsChunk, metricKey, maxResults, range, 'all');
        dispatch(action);
      });
    });
  }, [dispatch, maxResults, runUuids, metricKeys, range]);

  const refreshTimeoutRef = useRef<number | undefined>(undefined);
  const autoRefreshEnabledRef = useRef(autoRefreshEnabled && params.enabled);
  autoRefreshEnabledRef.current = autoRefreshEnabled && params.enabled;

  // Serialize runUuids to a string to use as a dependency in the effect,
  // directly used runUuids can cause unnecessary re-fetches
  const runUuidsSerialized = useMemo(() => runUuids.join(','), [runUuids]);

  // Regular single fetch effect with no auto-refresh capabilities. Used if auto-refresh is disabled.
  useEffect(() => {
    if (!enabled || autoRefreshEnabled) {
      return;
    }
    metricKeys.forEach((metricKey) => {
      chunk(runUuids, SAMPLED_METRIC_HISTORY_API_RUN_LIMIT).forEach((runUuidsChunk) => {
        const action = getSampledMetricHistoryBulkAction(runUuidsChunk, metricKey, maxResults, range);
        dispatch(action);
      });
    });
  }, [dispatch, maxResults, runUuids, metricKeys, range, enabled, autoRefreshEnabled]);

  // A fetch effect with auto-refresh capabilities. Used only if auto-refresh is enabled.
  useEffect(() => {
    let hookUnmounted = false;
    if (!enabled || !autoRefreshEnabled) {
      return;
    }

    // Base fetching function, used for both initial call and subsequent auto-refresh calls
    const fetchMetricsFn = async (isAutoRefreshing = false) => {
      const runUuids = runUuidsSerialized.split(',').filter((runUuid: string) => runUuid !== '');
      await Promise.all(
        metricKeys.map(async (metricKey) =>
          Promise.all(
            chunk(runUuids, SAMPLED_METRIC_HISTORY_API_RUN_LIMIT).map(async (runUuidsChunk) =>
              dispatch(
                getSampledMetricHistoryBulkAction(
                  runUuidsChunk,
                  metricKey,
                  maxResults,
                  range,
                  isAutoRefreshing ? 'auto' : undefined,
                ),
              ),
            ),
          ),
        ),
      );
    };

    const scheduleRefresh = async () => {
      // Initial check to confirm that auto-refresh is still enabled and the hook is still mounted
      if (!autoRefreshEnabledRef.current || hookUnmounted) {
        return;
      }
      try {
        await fetchMetricsFn(true);
      } catch (e) {
        // In case of error during auto-refresh, log the error but do break the auto-refresh loop
        Utils.logErrorAndNotifyUser(e);
      }
      clearTimeout(refreshTimeoutRef.current);

      // After loading the data, schedule the next refresh if the hook is still enabled and mounted
      if (!autoRefreshEnabledRef.current || hookUnmounted) {
        return;
      }

      refreshTimeoutRef.current = window.setTimeout(
        scheduleRefresh,
        EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL,
      );
    };

    fetchMetricsFn().then(scheduleRefresh);

    return () => {
      // Mark the hook as unmounted to prevent scheduling new auto-refreshes with current data
      hookUnmounted = true;

      // Clear the timeout
      clearTimeout(refreshTimeoutRef.current);
    };
  }, [dispatch, maxResults, runUuidsSerialized, metricKeys, range, enabled, autoRefreshEnabled]);

  return { isLoading, isRefreshing, resultsByRunUuid, refresh: refreshFn };
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
