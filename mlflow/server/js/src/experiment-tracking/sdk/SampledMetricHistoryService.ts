import { difference } from 'lodash';
import { getUUID } from '../../common/utils/ActionUtils';
import { fetchEndpoint, jsonBigIntResponseParser } from '../../common/utils/FetchUtils';
import type { AsyncAction, ReduxState, ThunkDispatch } from '../../redux-types';
import { createChartAxisRangeKey } from '../components/runs-charts/components/RunsCharts.common';
import type { MetricEntity } from '../types';
import { type ParsedQs, stringify as queryStringStringify } from 'qs';
import { EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL } from '../utils/MetricsUtils';

interface GetHistoryBulkIntervalResponseType {
  metrics: (MetricEntity & { run_id: string })[];
}

export const GET_SAMPLED_METRIC_HISTORY_API_BULK = 'GET_SAMPLED_METRIC_HISTORY_API_BULK';
export interface GetSampledMetricHistoryBulkAction
  extends AsyncAction<
    GetHistoryBulkIntervalResponseType,
    {
      id?: string;
      isRefreshing?: boolean;
      runUuids: string[];
      key: string;
      rangeKey: string;
      maxResults?: number;
    }
  > {
  type: 'GET_SAMPLED_METRIC_HISTORY_API_BULK';
}

export const getSampledMetricHistoryBulkAction =
  (
    runUuids: string[],
    metricKey: string,
    maxResults?: number,
    range?: [number | string, number | string],
    /**
     * Refresh mode.
     * If set to `all`, disregard cache and always fetch data for all run UUIDs.
     * If set to  `auto`, fetch data for run UUIDs that the data is considered stale.
     * If unset, fetch data for run UUIDs that we don't have data for.
     */
    refreshMode: 'all' | 'auto' | undefined = undefined,
  ) =>
  (dispatch: ThunkDispatch, getState: () => ReduxState) => {
    const rangeKey = createChartAxisRangeKey(range);
    const getExistingDataForRunUuid = (runUuid: string) =>
      getState().entities.sampledMetricsByRunUuid[runUuid]?.[metricKey];

    const skippedRunUuids = runUuids.filter((runUuid) => {
      // If refresh mode is set to `all`, no runs are skipped
      if (refreshMode === 'all') {
        return false;
      }
      const sampledHistoryEntry = getExistingDataForRunUuid(runUuid)?.[rangeKey];

      // If refresh mode is set to `auto`, skip runs that are fresh or are being loaded
      if (refreshMode === 'auto') {
        const timePassedSinceLastUpdate = Date.now() - (sampledHistoryEntry?.lastUpdatedTime || 0);
        const isFresh = timePassedSinceLastUpdate < EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL;
        const isInitialized = Boolean(sampledHistoryEntry?.lastUpdatedTime);
        const isLoadingOrRefreshing = sampledHistoryEntry?.loading || sampledHistoryEntry?.refreshing;

        // Skip loading data for runs that
        // - were not initialized before
        // - have fresh data
        // - are being loaded already
        return !isInitialized || isFresh || isLoadingOrRefreshing;
      }

      // If refresh mode is unset, skip runs that we already have data for
      return sampledHistoryEntry?.error || sampledHistoryEntry?.loading || sampledHistoryEntry?.metricsHistory;
    });

    const runUuidsToFetch = difference(runUuids, skippedRunUuids);

    if (!runUuidsToFetch.length || !decodeURIComponent(metricKey)) {
      return Promise.resolve();
    }

    // Prepare query params
    const queryParamsInput: ParsedQs = {
      run_ids: runUuidsToFetch,
      metric_key: decodeURIComponent(metricKey),
      max_results: maxResults?.toString(),
    };

    // Add range to query string if specified
    if (range) {
      const [start_step, end_step] = range;
      queryParamsInput['start_step'] = start_step.toString();
      queryParamsInput['end_step'] = end_step.toString();
    }

    // We are not using MlflowService because this endpoint requires
    // special query string treatment
    const queryParams = queryStringStringify(
      queryParamsInput,
      // This configures qs to stringify arrays as ?run_ids=123&run_ids=234
      { arrayFormat: 'repeat' },
    );

    const request = fetchEndpoint({
      relativeUrl: `ajax-api/2.0/mlflow/metrics/get-history-bulk-interval?${queryParams}`,
      success: jsonBigIntResponseParser,
    });

    return dispatch({
      type: GET_SAMPLED_METRIC_HISTORY_API_BULK,
      payload: request,
      meta: {
        id: getUUID(),
        runUuids: runUuidsToFetch,
        key: metricKey,
        rangeKey,
        maxResults,
        isRefreshing: Boolean(refreshMode),
      },
    });
  };
