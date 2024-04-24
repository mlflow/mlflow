import { chunk, keyBy } from 'lodash';
import { useCallback, useEffect, useMemo } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { ReduxState, ThunkDispatch } from '../../../../redux-types';
import { createChartAxisRangeKey } from '../components/RunsCharts.common';
import { getSampledMetricHistoryBulkAction } from '../../../sdk/SampledMetricHistoryService';
import { SampledMetricsByRunUuidState } from 'experiment-tracking/types';

type SampledMetricData = SampledMetricsByRunUuidState[string][string][string];

export type SampledMetricsByRun = {
  runUuid: string;
} & {
  [metricKey: string]: SampledMetricData;
};

const SAMPLED_METRIC_HISTORY_API_RUN_LIMIT = 100;
/**
 *
 * Automatically fetches sampled metric history for runs, used in run runs charts.
 * After updating list of metrics or runs, optimizes the request and fetches
 * only the missing entries.
 */
export const useSampledMetricHistory = (params: {
  runUuids: string[];
  metricKeys: string[];
  maxResults?: number;
  range?: [number, number];
  enabled?: boolean;
}) => {
  const { metricKeys, runUuids, enabled, maxResults, range } = params;
  const dispatch = useDispatch<ThunkDispatch>();

  const { resultsByRunUuid, isLoading, isRefreshing } = useSelector((store: ReduxState) => {
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
  });

  const refreshFn = useCallback(() => {
    metricKeys.forEach((metricKey) => {
      chunk(runUuids, SAMPLED_METRIC_HISTORY_API_RUN_LIMIT).forEach((runUuidsChunk) => {
        const action = getSampledMetricHistoryBulkAction(runUuidsChunk, metricKey, maxResults, range, true);
        dispatch(action);
      });
    });
  }, [dispatch, maxResults, runUuids, metricKeys, range]);

  useEffect(() => {
    // Skip if not disabled (e.g. chart is not visible)
    if (!enabled) {
      return;
    }

    metricKeys.forEach((metricKey) => {
      chunk(runUuids, SAMPLED_METRIC_HISTORY_API_RUN_LIMIT).forEach((runUuidsChunk) => {
        const action = getSampledMetricHistoryBulkAction(runUuidsChunk, metricKey, maxResults, range);
        dispatch(action);
      });
    });
  }, [dispatch, maxResults, runUuids, metricKeys, range, enabled]);

  return { isLoading, isRefreshing, resultsByRunUuid, refresh: refreshFn };
};
