import { groupBy } from 'lodash';
import { fulfilled, pending, rejected } from '../../common/utils/ActionUtils';
import type { AsyncFulfilledAction, AsyncPendingAction, AsyncRejectedAction } from '../../redux-types';
import {
  GET_SAMPLED_METRIC_HISTORY_API_BULK,
  type GetSampledMetricHistoryBulkAction,
} from '../sdk/SampledMetricHistoryService';
import type { SampledMetricsByRunUuidState } from '../types';

export const sampledMetricsByRunUuid = (
  state: SampledMetricsByRunUuidState = {},
  action:
    | AsyncFulfilledAction<GetSampledMetricHistoryBulkAction>
    | AsyncPendingAction<GetSampledMetricHistoryBulkAction>
    | AsyncRejectedAction<GetSampledMetricHistoryBulkAction>,
) => {
  if (action.type === rejected(GET_SAMPLED_METRIC_HISTORY_API_BULK) && action.meta) {
    const { runUuids, key, rangeKey } = action.meta;
    const updatedState = { ...state };
    for (const runUuid of runUuids) {
      if (updatedState[runUuid]?.[key]?.[rangeKey]) {
        const existingEntry = updatedState[runUuid][key][rangeKey];
        updatedState[runUuid][key][rangeKey] = {
          // In case of failure, retain previous data entry and set error
          ...existingEntry,
          error: action.payload,
          refreshing: false,
          loading: false,
        };
      }
    }
    return updatedState;
  }
  if (action.type === pending(GET_SAMPLED_METRIC_HISTORY_API_BULK) && action.meta) {
    const { runUuids, key, rangeKey, isRefreshing } = action.meta;
    const updatedState = { ...state };
    for (const runUuid of runUuids) {
      if (!updatedState[runUuid]) {
        updatedState[runUuid] = {
          [key]: {
            [rangeKey]: {
              metricsHistory: undefined,
              refreshing: false,
              loading: true,
            },
          },
        };
      } else if (!updatedState[runUuid][key]) {
        updatedState[runUuid][key] = {
          [rangeKey]: {
            metricsHistory: undefined,
            refreshing: false,
            loading: true,
          },
        };
      } else if (!updatedState[runUuid][key][rangeKey]) {
        updatedState[runUuid][key][rangeKey] = {
          metricsHistory: undefined,
          refreshing: false,
          loading: true,
        };
      } else if (updatedState[runUuid][key][rangeKey] && isRefreshing) {
        updatedState[runUuid][key][rangeKey] = {
          ...updatedState[runUuid][key][rangeKey],
          refreshing: true,
        };
      }
    }
    return updatedState;
  }
  if (action.type === fulfilled(GET_SAMPLED_METRIC_HISTORY_API_BULK) && action.meta) {
    const { runUuids, key, rangeKey } = action.meta;

    const updatedState = { ...state };
    const { metrics } = action.payload;
    const resultsByRunUuid = groupBy(metrics, 'run_id');

    for (const runUuid of runUuids) {
      const resultList = resultsByRunUuid[runUuid];
      if (updatedState[runUuid]?.[key]?.[rangeKey]) {
        updatedState[runUuid][key][rangeKey] = {
          metricsHistory: resultList || [],
          loading: false,
          refreshing: false,
          lastUpdatedTime: Date.now(),
        };
      }
    }
    return updatedState;
  }
  return state;
};
