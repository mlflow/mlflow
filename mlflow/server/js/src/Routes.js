import {X_AXIS_RELATIVE} from "./components/MetricsPlotControls";

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

  static getMetricPageRoute(runUuids, metricKey, experimentId, plotMetricKeys = null,
                            plotLayout = {}, selectedXAxis = X_AXIS_RELATIVE, yAxisLogScale = false,
                            lineSmoothness = 0, showPoint = false, selectedRunIds = null) {
    // If runs to display are specified (e.g. if user filtered to specific runs in a metric
    // comparison plot), embed them in the URL, otherwise default to metricKey
    const finalPlotMetricKeys = plotMetricKeys || [metricKey];
    const initialSelectedRunIds = selectedRunIds ||
        [...Array(runUuids.length * finalPlotMetricKeys.length).keys()];
    // Convert boolean to enum to keep URL format extensible to adding new types of y axis scales
    const yAxisScale = yAxisLogScale ? "log" : "linear";
    return `/metric/${encodeURIComponent(metricKey)}?runs=${JSON.stringify(runUuids)}&` +
      `experiment=${experimentId}` +
      `&plot_metric_keys=${JSON.stringify(finalPlotMetricKeys)}` +
      `&plot_layout=${JSON.stringify(plotLayout)}` +
      `&x_axis=${selectedXAxis}` +
      `&y_axis_scale=${yAxisScale}` +
      `&line_smoothness=${lineSmoothness}` +
      `&show_point=${showPoint}` +
      `&selected_run_ids=${JSON.stringify(initialSelectedRunIds)}`;
  }

  static metricPageRoute = "/metric/:metricKey";

  static getCompareRunPageRoute(runUuids, experimentId) {
    return `/compare-runs?runs=${JSON.stringify(runUuids)}&experiment=${experimentId}`;
  }

  static compareRunPageRoute = "/compare-runs"
}

export default Routes;
