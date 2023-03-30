import { useCallback, useEffect, useMemo, useState } from 'react';
import { getMetricHistoryApiBulk } from '../../../actions';
import type { MetricHistoryByName } from '../../../types';
import { useAsyncDispatch } from '../../experiment-page/hooks/useAsyncDispatch';
import type { CompareChartRunData } from '../charts/CompareRunsCharts.common';

/**
 * Automatically fetches metric history for runs, used in compare runs charts.
 * After updating list of metrics or runs, optimizes the request and fetches
 * only the missing entries.
 */
export const useFetchCompareRunsMetricHistory = (
  // We can fetch multiple metrics at once
  metricKeys: string[],
  runsData: CompareChartRunData[],
  currentStoreMetrics: Record<string, MetricHistoryByName>,
) => {
  const dispatch = useAsyncDispatch();

  const [error, setError] = useState<any>(null);
  const [requests, setRequests] = useState<Record<string, boolean>>({});

  /**
   * We store pending requests in "<run-id>-<metric-key>" format,
   * the function below changes the state of pending requests
   */
  const setRequestsPending = useCallback(
    (runUuids: string[], metric: string, value: boolean) => {
      setRequests((currentRequests) => {
        const result = { ...currentRequests };
        for (const uuid of runUuids) {
          const key = `${uuid}-${metric}`;
          result[key] = value;
        }
        return result;
      });
    },
    [setRequests],
  );

  const addRequests = useCallback(
    (runUuids: string[], metric: string) => setRequestsPending(runUuids, metric, true),
    [setRequestsPending],
  );

  const settleRequests = useCallback(
    (runUuids: string[], metric: string) => setRequestsPending(runUuids, metric, false),
    [setRequestsPending],
  );

  const isLoading = useMemo(() => {
    const runUuids = runsData.map((r) => r.runInfo.run_uuid);
    for (const uuid of runUuids) {
      for (const metric of metricKeys) {
        const isPendingRequest = requests[`${uuid}-${metric}`];
        if (isPendingRequest) {
          return true;
        }
      }
    }
    return false;
  }, [metricKeys, requests, runsData]);

  useEffect(() => {
    if (!metricKeys.length) {
      return;
    }

    for (const metricKey of metricKeys) {
      if (!metricKey) {
        continue;
      }

      // Determine which runs does not have corresponding
      // metric history entries already fetched and stored
      const runUuids = runsData.map((r) => r.runInfo.run_uuid);
      const runUuidsToFetch = runUuids.filter((runUuid) => {
        const isInStore = Boolean(currentStoreMetrics[runUuid]?.[metricKey]);
        const isPendingRequest = requests[`${runUuid}-${metricKey}`];
        return !isInStore && !isPendingRequest;
      });

      if (!runUuidsToFetch.length) {
        continue;
      }

      // Register request for history in the internal state
      addRequests(runUuidsToFetch, metricKey);

      // Dispatch the action
      dispatch(getMetricHistoryApiBulk(runUuidsToFetch, metricKey))
        .then(() => {
          // Settle request in the internal state if it's resolved
          settleRequests(runUuidsToFetch, metricKey);
        })
        .catch((e) => {
          // Set the internal state error if occurred
          setError(e);
        });
    }
  }, [addRequests, currentStoreMetrics, dispatch, settleRequests, metricKeys, requests, runsData]);

  return { isLoading, error };
};
