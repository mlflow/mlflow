import { X_AXIS_RELATIVE } from './components/MetricsPlotControls';

class Routes {
  static rootRoute = '/';

  static getExperimentPageRoute(experimentId) {
    return `/experiments/${experimentId}`;
  }

  static experimentPageRoute = '/experiments/:experimentId';

  static experimentPageSearchRoute = '/experiments/:experimentId/:searchString';

  static getRunPageRoute(experimentId, runUuid) {
    return `/experiments/${experimentId}/runs/${runUuid}`;
  }

  static getRunArtifactRoute(experimentId, runUuid, artifactPath) {
    return `${this.getRunPageRoute(experimentId, runUuid)}/artifactPath/${artifactPath}`;
  }

  static runPageRoute = '/experiments/:experimentId/runs/:runUuid';

  static runPageWithArtifactSelectedRoute =
    '/experiments/:experimentId/runs/:runUuid/artifactPath/:initialSelectedArtifactPath+';

  /**
   * Get route to the metric plot page
   * @param runUuids - Array of string run IDs to plot
   * @param metricKey - Primary metric key in plot, shown in page breadcrumb
   * @param experimentId - ID of experiment to link to from page breadcrumb
   * @param plotMetricKeys - Array of string metric keys to plot
   * @param plotLayout - Object containing plot layout information in Plotly format. See
   *   https://plot.ly/javascript/plotlyjs-events/#update-data for an idea of object structure
   * @param selectedXAxis - Enum (string) describing type of X axis (wall time, relative time, step)
   * @param yAxisLogScale - Boolean - if true, y axis should be displayed on a log scale
   *   (y axis scale is assumed to be linear otherwise)
   * @param lineSmoothness - Float, coefficient >= 0 describing how much line smoothing to apply
   * @param showPoint - Boolean, whether or not to show dots at individual data points in the metric
   *   line plot
   * @param deselectedCurves - Array of strings where each string describes a curve that was
   *   deselected / toggled off by the user (a curve that should not be displayed in the metric
   *   plot). Strings are of the form "<runId>-<metricKey>". We describe the plot in terms
   *   of deselected curves as we don't know a-priori which runs from
   *   runUuids contain which of the metric keys in plotMetricKeys
   * @param lastLinearYAxisRange - Array containing most recent bounds of a linear-scale y axis.
   *   Used to keep track of the most-recent linear y-axis plot range, to handle the specific
   *   case where we toggle a plot with negative y-axis bounds from linear to log scale,
   *   and then back to linear scale (we save the initial negative linear y-axis bounds so
   *   that we can restore them when converting from log back to linear scale)
   * @returns {string}
   */
  static getMetricPageRoute(
    runUuids,
    metricKey,
    experimentId,
    plotMetricKeys = null,
    plotLayout = {},
    selectedXAxis = X_AXIS_RELATIVE,
    yAxisLogScale = false,
    lineSmoothness = 1,
    showPoint = false,
    deselectedCurves = [],
    lastLinearYAxisRange = [],
  ) {
    // If runs to display are specified (e.g. if user filtered to specific runs in a metric
    // comparison plot), embed them in the URL, otherwise default to metricKey
    const finalPlotMetricKeys = plotMetricKeys || [metricKey];
    // Convert boolean to enum to keep URL format extensible to adding new types of y axis scales
    const yAxisScale = yAxisLogScale ? 'log' : 'linear';
    return (
      `/metric/${encodeURIComponent(metricKey)}?runs=${JSON.stringify(runUuids)}&` +
      `experiment=${experimentId}` +
      `&plot_metric_keys=${JSON.stringify(finalPlotMetricKeys)}` +
      `&plot_layout=${JSON.stringify(plotLayout)}` +
      `&x_axis=${selectedXAxis}` +
      `&y_axis_scale=${yAxisScale}` +
      `&line_smoothness=${lineSmoothness}` +
      `&show_point=${showPoint}` +
      `&deselected_curves=${JSON.stringify(deselectedCurves)}` +
      `&last_linear_y_axis_range=${JSON.stringify(lastLinearYAxisRange)}`
    );
  }

  static metricPageRoute = '/metric/:metricKey';

  static getCompareRunPageRoute(runUuids, experimentId) {
    return `/compare-runs?runs=${JSON.stringify(runUuids)}&experiment=${experimentId}`;
  }

  static compareRunPageRoute = '/compare-runs';
}

export default Routes;
