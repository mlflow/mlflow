/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { minBy, maxBy } from 'lodash';
import {
  GET_METRIC_HISTORY_API,
  GET_METRIC_HISTORY_API_BULK,
  GET_RUN_API,
  LOAD_MORE_RUNS_API,
  SEARCH_RUNS_API,
} from '../actions';
import { Metric } from '../sdk/MlflowMessages';
import { fulfilled } from '../../common/utils/ActionUtils';

export const getMetricsByKey = (runUuid: any, key: any, state: any) => {
  return state.entities.metricsByRunUuid[runUuid][key];
};

/**
 * Return
 * {
 *   [metric.key]: metric
 *   ...
 * }, one per metricName
 */
export const getLatestMetrics = (runUuid: any, state: any) => {
  return state.entities.latestMetricsByRunUuid[runUuid];
};

export const getMinMetrics = (runUuid: any, state: any) => {
  return state.entities.minMetricsByRunUuid[runUuid];
};

export const getMaxMetrics = (runUuid: any, state: any) => {
  return state.entities.maxMetricsByRunUuid[runUuid];
};

/**
 * Return latest metrics by run UUID (object of run UUID -> object of metric key -> Metric object)
 */
export const latestMetricsByRunUuid = (state = {}, action: any) => {
  const metricArrToObject = (metrics: any) => {
    const metricObj = {};
    // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
    metrics.forEach((m: any) => (metricObj[m.key] = (Metric as any).fromJs(m)));
    return metricObj;
  };
  switch (action.type) {
    case fulfilled(GET_RUN_API): {
      const runInfo = action.payload.run.info;
      const runUuid = runInfo.runUuid;
      const metrics = action.payload.run.data.metrics || [];
      return {
        ...state,
        [runUuid]: metricArrToObject(metrics),
      };
    }
    case fulfilled(SEARCH_RUNS_API):
    case fulfilled(LOAD_MORE_RUNS_API): {
      const newState = { ...state };
      if (action.payload.runs) {
        action.payload.runs.forEach((rJson: any) => {
          const runUuid = rJson.info.runUuid;
          const metrics = rJson.data.metrics || [];
          // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
          newState[runUuid] = metricArrToObject(metrics);
        });
      }
      return newState;
    }
    case fulfilled(GET_METRIC_HISTORY_API): {
      const newState = { ...state };
      const { runUuid, key } = action.meta;
      const { metrics } = action.payload;
      if (metrics && metrics.length > 0) {
        const lastMetric = (Metric as any).fromJs(metrics[metrics.length - 1]);
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        if (newState[runUuid]) {
          // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
          newState[runUuid][key] = lastMetric;
        } else {
          // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
          newState[runUuid] = { [key]: lastMetric };
        }
      }
      return newState;
    }
    case fulfilled(GET_METRIC_HISTORY_API_BULK): {
      const newState = { ...state };
      const { runUuids, key } = action.meta;
      const { metrics } = action.payload;
      if (metrics && metrics.length > 0) {
        for (const runUuid of runUuids) {
          const runMetrics = metrics.filter((m: any) => m.run_id === runUuid);
          if (runMetrics.length < 1) {
            continue;
          }
          const lastMetric = (Metric as any).fromJs(runMetrics[runMetrics.length - 1]);
          // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
          if (newState[runUuid]) {
            // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
            newState[runUuid][key] = lastMetric;
          } else {
            // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
            newState[runUuid] = { [key]: lastMetric };
          }
        }
      }
      return newState;
    }
    default:
      return state;
  }
};

const reducedMetricsByRunUuid = (state = {}, action: any, reducer: any) => {
  switch (action.type) {
    case fulfilled(GET_METRIC_HISTORY_API): {
      const newState = { ...state };
      const { runUuid, key } = action.meta;
      const { metrics } = action.payload;
      if (metrics && metrics.length > 0) {
        const reducedMetric = (Metric as any).fromJs(reducer(metrics));
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        if (newState[runUuid]) {
          // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
          newState[runUuid][key] = reducedMetric;
        } else {
          // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
          newState[runUuid] = { [key]: reducedMetric };
        }
      }
      return newState;
    }
    case fulfilled(GET_METRIC_HISTORY_API_BULK): {
      const newState = { ...state };
      const { runUuids, key } = action.meta;
      const { metrics } = action.payload;
      if (metrics && metrics.length > 0) {
        for (const runUuid of runUuids) {
          const runMetrics = metrics.filter((m: any) => m.run_id === runUuid);
          const reducerResult = reducer(runMetrics);
          if (!reducerResult) {
            continue;
          }
          const reducedMetric = (Metric as any).fromJs(reducerResult);
          // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
          if (newState[runUuid]) {
            // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
            newState[runUuid][key] = reducedMetric;
          } else {
            // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
            newState[runUuid] = { [key]: reducedMetric };
          }
        }
      }
      return newState;
    }
    default:
      return state;
  }
};

/**
 * Return minimum metrics by run UUID (object of run UUID -> object of metric key -> Metric object)
 */
export const minMetricsByRunUuid = (state = {}, action: any) =>
  reducedMetricsByRunUuid(state, action, (metrics: any) => minBy(metrics, 'value'));

/**
 * Return maximum metrics by run UUID (object of run UUID -> object of metric key -> Metric object)
 */
export const maxMetricsByRunUuid = (state = {}, action: any) =>
  reducedMetricsByRunUuid(state, action, (metrics: any) => maxBy(metrics, 'value'));

export const metricsByRunUuid = (state = {}, action: any) => {
  switch (action.type) {
    case fulfilled(GET_METRIC_HISTORY_API): {
      const { runUuid } = action.meta;
      const metrics = action.payload.metrics || [];
      return {
        ...state,
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        [runUuid]: metricsByKey(state[runUuid], action, metrics),
      };
    }
    case fulfilled(GET_METRIC_HISTORY_API_BULK): {
      const { runUuids } = action.meta;
      const metrics = action.payload.metrics || [];
      const newState = { ...state };

      for (const runUuid of runUuids) {
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        newState[runUuid] = metricsByKey(
          // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
          state[runUuid],
          action,
          metrics.filter((m: any) => m.run_id === runUuid),
        );
      }

      return newState;
    }
    default:
      return state;
  }
};

export const metricsByKey = (state = {}, action: any, metrics: any) => {
  const newState = { ...state };
  switch (action.type) {
    case fulfilled(GET_METRIC_HISTORY_API):
    case fulfilled(GET_METRIC_HISTORY_API_BULK): {
      const { key, pageToken } = action.meta;
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      const existingMetrics = newState[key] || [];
      const newMetrics = metrics.map((m: any) => (Metric as any).fromJs(m));
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      newState[key] = pageToken ? [...existingMetrics, ...newMetrics] : newMetrics;
      return newState;
    }
    default:
      return state;
  }
};
