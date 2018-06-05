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
  static getMetricPageRoute(runUuids, metricKey) {
    return `/metric/${metricKey}?runs=${JSON.stringify(runUuids)}`;
  }
  static metricPageRoute = "/metric/:metricKey";

  static getCompareRunPageRoute(runUuids) {
    return `/compare-runs?runs=${JSON.stringify(runUuids)}`
  }
  static compareRunPageRoute = "/compare-runs"
}

export default Routes;