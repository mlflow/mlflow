import { difference } from 'lodash';
import { getUUID } from '../../common/utils/ActionUtils';
import { fetchEndpoint, jsonBigIntResponseParser } from '../../common/utils/FetchUtils';
import { AsyncAction, ReduxState, ThunkDispatch } from '../../redux-types';
import { createChartAxisRangeKey } from '../components/runs-charts/components/RunsCharts.common';
import { MetricEntity } from '../types';
import { type ParsedQs, stringify as queryStringStringify } from 'qs';

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
    isRefreshing = false,
  ) =>
  (dispatch: ThunkDispatch, getState: () => ReduxState) => {
    const rangeKey = createChartAxisRangeKey(range);
    const getExistingDataForRunUuid = (runUuid: string) =>
      getState().entities.sampledMetricsByRunUuid[runUuid]?.[metricKey];

    // Find run UUIDs that we already have data for
    const skippedRunUuids = runUuids.filter(
      (runUuid) =>
        getExistingDataForRunUuid(runUuid)?.[rangeKey]?.error ||
        getExistingDataForRunUuid(runUuid)?.[rangeKey]?.loading ||
        getExistingDataForRunUuid(runUuid)?.[rangeKey]?.metricsHistory,
    );

    // If we are not refreshing, use only run UUIDs that we don't have data for.
    // If we are refreshing, fetch data for all run UUIDs.
    const runUuidsToFetch = isRefreshing ? runUuids : difference(runUuids, skippedRunUuids);

    if (!runUuidsToFetch.length || !decodeURIComponent(metricKey)) {
      return;
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

    dispatch({
      type: GET_SAMPLED_METRIC_HISTORY_API_BULK,
      payload: request,
      meta: {
        id: getUUID(),
        runUuids: runUuidsToFetch,
        key: metricKey,
        rangeKey,
        maxResults,
        isRefreshing,
      },
    });
  };
