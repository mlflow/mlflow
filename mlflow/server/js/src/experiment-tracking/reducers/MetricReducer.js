import {
  GET_METRIC_HISTORY_API,
  GET_RUN_API,
  LOAD_MORE_RUNS_API,
  SEARCH_RUNS_API,
} from '../actions';
import { RunInfo, Metric } from '../sdk/MlflowMessages';
import { fulfilled } from '../../common/utils/ActionUtils';

export const getMetricsByKey = (runUuid, key, state) => {
  return state.entities.metricsByRunUuid[runUuid][key];
};

/**
 * Return
 * {
 *   [metric.key]: metric
 *   ...
 * }, one per metricName
 */
export const getLatestMetrics = (runUuid, state) => {
  return state.entities.latestMetricsByRunUuid[runUuid];
};

/**
 * Return latest metrics by run UUID (object of run UUID -> object of metric key -> Metric object)
 */
export const latestMetricsByRunUuid = (state = {}, action) => {
  const metricArrToObject = (metrics) => {
    const metricObj = {};
    metrics.forEach((m) => (metricObj[m.key] = Metric.fromJs(m)));
    return metricObj;
  };
  switch (action.type) {
    case fulfilled(GET_RUN_API): {
      const runInfo = RunInfo.fromJs(action.payload.run.info);
      const runUuid = runInfo.getRunUuid();
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
        action.payload.runs.forEach((rJson) => {
          const runUuid = rJson.info.run_uuid;
          const metrics = rJson.data.metrics || [];
          newState[runUuid] = metricArrToObject(metrics);
        });
      }
      return newState;
    }
    case fulfilled(GET_METRIC_HISTORY_API): {
      const newState = { ...state };
      const { runUuid, key } = action.meta.runUuid;
      const { metrics } = action.payload;
      if (metrics && metrics.length > 0) {
        const lastMetric = Metric.fromJs(metrics[metrics.length - 1]);
        if (newState[runUuid]) {
          newState[runUuid][key] = lastMetric;
        } else {
          newState[runUuid] = { [key]: lastMetric };
        }
      }
      return newState;
    }
    default:
      return state;
  }
};

export const metricsByRunUuid = (state = {}, action) => {
  switch (action.type) {
    case fulfilled(GET_METRIC_HISTORY_API): {
      const { runUuid } = action.meta;
      const metrics = action.payload.metrics || [];
      return {
        ...state,
        [runUuid]: metricsByKey(state[runUuid], action, metrics),
      };
    }
    default:
      return state;
  }
};

export const metricsByKey = (state = {}, action, metrics) => {
  const newState = { ...state };
  switch (action.type) {
    case fulfilled(GET_METRIC_HISTORY_API): {
      const { key } = action.meta;
      newState[key] = metrics.map((m) => Metric.fromJs(m));
      return newState;
    }
    default:
      return state;
  }
};
