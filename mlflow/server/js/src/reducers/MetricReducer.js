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
      const newState = {
        ...state,
         [runUuid]: metricsByKey(state[runUuid], action, metrics)
      };
      console.log("GetRun meta: " + JSON.stringify(action.meta));
      console.log("GetRun: metrics for run " + runUuid + ": " + JSON.stringify(newState[runUuid]));
      return newState;
    }
    case fulfilled(GET_METRIC_HISTORY_API): {
      const runUuid = action.meta.runUuid;
      const metrics = action.payload.metrics || [];
      const newState = {
        ...state,
        [runUuid]: metricsByKey(state[runUuid], action, metrics)
      };
      console.log("GetMetricHistory meta: " + JSON.stringify(action.meta));
      const key = action.meta.key;
      const length = newState[runUuid][key].size;
      console.log("GetMetricHistory: metrics for run " + runUuid + ", metric key: " + key + ", length: " + length);
      return newState;
    }
    case fulfilled(SEARCH_RUNS_API): {
      const newState = { ...state };
      if (action.payload.runs) {
        action.payload.runs.forEach((rJson) => {
          const run = Run.fromJs(rJson);
          const runUuid = run.getInfo().getRunUuid();
          const metrics = rJson.data.metrics || [];
          newState[runUuid] = metricsByKey(newState[runUuid], action, metrics);
          console.log("SearchRuns meta: " + JSON.stringify(action.meta));
          console.log("SearchRuns: metrics for run " + runUuid + ": " + JSON.stringify(newState[runUuid]));
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
    // Assumes the GET_RUN_API only returns 1 metric per key.
    case fulfilled(GET_RUN_API): {
      metrics.forEach((m) => {
        if (newState[m.key]) {
          newState[m.key].add(Metric.fromJs(m));
        } else {
          newState[m.key] = new Set([Metric.fromJs(m)]);
        }
      });
      return newState;
    }
    // Assumes the SEARCH_RUNS_API only returns 1 metric per key.
    case fulfilled(SEARCH_RUNS_API): {
      metrics.forEach((m) => {
        if (newState[m.key]) {
          newState[m.key].add(Metric.fromJs(m));
        } else {
          newState[m.key] = new Set([Metric.fromJs(m)]);
        }
      });
      return newState;
    }
    case fulfilled(GET_METRIC_HISTORY_API): {
      const key = action.meta.key;
      newState[key] = new Set(metrics.map((m) => Metric.fromJs(m)));
      return newState;
    }
    default:
      return state;
  }
};
