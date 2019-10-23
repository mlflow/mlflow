class Routes {
  static rootRoute = "/";

  static getExperimentPageRoute(experimentId) {
    return `/experiments/${experimentId}`;
  }

  static experimentPageRoute = "/experiments/:experimentId";

  static experimentPageSearchRoute = "/experiments/:experimentId/:searchString";

  static getRunPageRoute(experimentId, runUuid) {
    return `/experiments/${experimentId}/runs/${runUuid}`;
  }

  static runPageRoute = "/experiments/:experimentId/runs/:runUuid";

  static getMetricPageRoute(runUuids, metricKey, experimentId, plotMetricKeys) {
    return `/metric/${encodeURIComponent(metricKey)}?runs=${JSON.stringify(runUuids)}&` +
      `experiment=${experimentId}` +
      `&plot_metric_keys=${JSON.stringify(plotMetricKeys || [metricKey])}`;
  }

  static metricPageRoute = "/metric/:metricKey";

  static getCompareRunPageRoute(runUuids, experimentId) {
    return `/compare-runs?runs=${JSON.stringify(runUuids)}&experiment=${experimentId}`;
  }

  static compareRunPageRoute = "/compare-runs"
}

export default Routes;
