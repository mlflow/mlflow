class Routes {
  static rootRoute = "/";

  static getExperimentPageRoute(experimentId) {
    return `/experiments/${experimentId}`;
  }

  static experimentPageRoute = "/experiments/:experimentId";

  static getRunPageRoute(experimentId, runUuid) {
    return `/experiments/${experimentId}/runs/${runUuid}`;
  }

  static runPageRoute = "/experiments/:experimentId/runs/:runUuid";

  static getMetricPageRoute(runUuids, metricKey, experimentId) {
    return `/metric/${metricKey}?runs=${JSON.stringify(runUuids)}&experiment=${experimentId}`;
  }

  static metricPageRoute = "/metric/:metricKey";

  static getCompareRunPageRoute(runUuids, experimentId) {
    return `/compare-runs?runs=${JSON.stringify(runUuids)}&experiment=${experimentId}`;
  }

  static compareRunPageRoute = "/compare-runs"
}

export default Routes;
