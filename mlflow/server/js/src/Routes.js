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
    let route = `/metric/${metricKey}?runs=${JSON.stringify(runUuids)}`;
    if (experimentId !== undefined) {
      route += `&experiment=${experimentId}`;
    }
    return route;
  }
  static metricPageRoute = "/metric/:metricKey";

  static getCompareRunPageRoute(runUuids, experimentId) {
    let route = `/compare-runs?runs=${JSON.stringify(runUuids)}`;
    if (experimentId !== undefined) {
      route += `&experiment=${experimentId}`;
    }
    return route;
  }
  static compareRunPageRoute = "/compare-runs"
}

export default Routes;
