import { fulfilled, GET_METRIC_HISTORY_API, GET_RUN_API, SEARCH_RUNS_API } from '../Actions';
import { Run, RunInfo, Metric } from '../sdk/MlflowMessages';

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
  const metricsByKey = state.entities.metricsByRunUuid[runUuid];
  if (!metricsByKey) {
    return {};
  }
  let ret = {};
  Object.values(metricsByKey).forEach((metricsForKey) => {
    // metricsForKey should be an array with length always greater than 0.
    let lastMetric;
    metricsForKey.forEach((metric) => {
      if (lastMetric === undefined || lastMetric.getTimestamp() <= metric.getTimestamp()) {
        lastMetric = metric;
      }
    });
    ret = {
      ...ret,
      [lastMetric.getKey()]: lastMetric
    };
  });
  return ret;
};


export const metricsByRunUuid = (state = {}, action) => {
  switch (action.type) {
    case fulfilled(GET_RUN_API): {
      const runInfo = RunInfo.fromJs(action.payload.run.info);
      const runUuid = runInfo.getRunUuid();
      const metrics = action.payload.run.data.metrics || [];
      return {
        ...state,
         [runUuid]: metricsByKey(state[runUuid], action, metrics)
      };
    }
    case fulfilled(GET_METRIC_HISTORY_API): {
      const runUuid = action.meta.runUuid;
      const metrics = action.payload.metrics || [];
      return {
        ...state,
        [runUuid]: metricsByKey(state[runUuid], action, metrics)
      };
    }
    case fulfilled(SEARCH_RUNS_API): {
      const newState = { ...state };
      if (action.payload.runs) {
        action.payload.runs.forEach((rJson) => {
          const run = Run.fromJs(rJson);
          const runUuid = run.getInfo().getRunUuid();
          const metrics = rJson.data.metrics || [];
          newState[runUuid] = metricsByKey(newState[runUuid], action, metrics);
        });
      }
      return newState;
    }
    default:
      return state;
  }
};

const metricsByKey = (state = {}, action, metrics) => {
  const newState = { ...state };
  switch (action.type) {
    // Assumes the GET_RUN_API only returns 1 metric (the latest metric) for each key.
    case fulfilled(GET_RUN_API): {
      metrics.forEach((m) => {
        const newArr = newState[m.key] ? newState[m.key].slice(0, newState[m.key].length - 1) : []
        newArr.push(Metric.fromJs(m));
        newState[m.key] = newArr;
      });
      return newState;
    }
    // Assumes the SEARCH_RUNS_API only returns 1 metric (the latest metric) per key.
    case fulfilled(SEARCH_RUNS_API): {
      metrics.forEach((m) => {
        const newArr = newState[m.key] ? newState[m.key].slice(0, newState[m.key].length - 1) : []
        newArr.push(Metric.fromJs(m));
        newState[m.key] = newArr;
      });
      return newState;
    }
    case fulfilled(GET_METRIC_HISTORY_API): {
      const key = action.meta.key;
      newState[key] = metrics.map((m) => Metric.fromJs(m));
      return newState;
    }
    default:
      return state;
  }
};
