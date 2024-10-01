import { createMLflowRoutePath, generatePath } from '../common/utils/RoutingUtils';

// Route path definitions (used in defining route elements)
export class RoutePaths {
  static get rootRoute() {
    return createMLflowRoutePath('/');
  }
  static get experimentObservatory() {
    return createMLflowRoutePath('/experiments');
  }
  static get experimentPage() {
    return createMLflowRoutePath('/experiments/:experimentId');
  }
  static get experimentPageSearch() {
    return createMLflowRoutePath('/experiments/:experimentId/:searchString');
  }
  static get runPage() {
    return createMLflowRoutePath('/experiments/:experimentId/runs/:runUuid');
  }
  // More flexible route path, supporting subpages (tabs) and multi-slash artifact paths.
  // Will eventually replace "runPage" above.
  static get runPageWithTab() {
    return createMLflowRoutePath('/experiments/:experimentId/runs/:runUuid/*');
  }
  static get runPageWithArtifact() {
    return createMLflowRoutePath('/experiments/:experimentId/runs/:runUuid/artifactPath/*');
  }
  static get runPageDirect() {
    return createMLflowRoutePath('/runs/:runUuid');
  }
  static get metricPage() {
    return createMLflowRoutePath('/metric/*');
  }
  static get compareRuns() {
    return createMLflowRoutePath('/compare-runs');
  }
  static get compareExperiments() {
    return createMLflowRoutePath('/compare-experiments');
  }
  static get compareExperimentsSearch() {
    return createMLflowRoutePath('/compare-experiments/:searchString');
  }
}

// Concrete routes and functions for generating parametrized paths
class Routes {
  static get rootRoute() {
    return RoutePaths.rootRoute;
  }

  static get experimentsObservatoryRoute() {
    return RoutePaths.experimentObservatory;
  }

  static get experimentPageRoute() {
    return RoutePaths.experimentPage;
  }

  static get experimentPageSearchRoute() {
    return RoutePaths.experimentPageSearch;
  }

  static getExperimentPageRoute(experimentId: string, isComparingRuns = false, shareState?: string) {
    const path = generatePath(RoutePaths.experimentPage, { experimentId });
    if (shareState) {
      return `${path}?viewStateShareKey=${shareState}`;
    }
    if (isComparingRuns) {
      return `${path}?isComparingRuns=true`;
    }
    return path;
  }

  static searchRunsByUser(experimentId: string, userId: string) {
    const path = generatePath(RoutePaths.experimentPage, { experimentId });
    const filterString = `user_id = '${userId}'`;
    return `${path}?searchFilter=${encodeURIComponent(filterString)}`;
  }

  static searchRunsByLifecycleStage(experimentId: string, lifecycleStage: string) {
    const path = generatePath(RoutePaths.experimentPage, { experimentId });
    return `${path}?lifecycleFilter=${lifecycleStage}`;
  }

  static getRunPageRoute(experimentId: string, runUuid: string, artifactPath: string | null = null) {
    if (artifactPath) {
      return this.getRunPageTabRoute(experimentId, runUuid, ['artifacts', artifactPath].join('/'));
    }
    return generatePath(RoutePaths.runPage, { experimentId, runUuid });
  }

  static getRunPageTabRoute(experimentId: string, runUuid: string, tabPath?: string) {
    return generatePath(RoutePaths.runPageWithTab, {
      experimentId,
      runUuid,
      '*': tabPath,
    });
  }

  /**
   * Get route to the metric plot page
   * @param runUuids - Array of string run IDs to plot
   * @param metricKey - Primary metric key in plot, shown in page breadcrumb
   * @param experimentIds - IDs of experiments to link to from page breadcrumb
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
   */
  static getMetricPageRoute(
    runUuids: string[],
    metricKey: string,
    experimentIds: string[],
    plotMetricKeys: string[] | null = null,
    plotLayout: any = {},
    selectedXAxis: 'wall' | 'step' | 'relative' = 'relative',
    yAxisLogScale = false,
    lineSmoothness = 1,
    showPoint = false,
    deselectedCurves: string[] = [],
    lastLinearYAxisRange: string[] = [],
  ) {
    // If runs to display are specified (e.g. if user filtered to specific runs in a metric
    // comparison plot), embed them in the URL, otherwise default to metricKey
    const finalPlotMetricKeys = plotMetricKeys || [metricKey];
    // Convert boolean to enum to keep URL format extensible to adding new types of y axis scales
    const yAxisScale = yAxisLogScale ? 'log' : 'linear';

    const queryString =
      `?runs=${JSON.stringify(runUuids)}` +
      `&metric=${encodeURIComponent(JSON.stringify(metricKey))}` +
      `&experiments=${JSON.stringify(experimentIds)}` +
      `&plot_metric_keys=${encodeURIComponent(JSON.stringify(finalPlotMetricKeys))}` +
      `&plot_layout=${JSON.stringify(plotLayout)}` +
      `&x_axis=${selectedXAxis}` +
      `&y_axis_scale=${yAxisScale}` +
      `&line_smoothness=${lineSmoothness}` +
      `&show_point=${showPoint}` +
      `&deselected_curves=${JSON.stringify(deselectedCurves)}` +
      `&last_linear_y_axis_range=${JSON.stringify(lastLinearYAxisRange)}`;

    return `${generatePath(RoutePaths.metricPage)}${queryString}`;
  }

  static getCompareRunPageRoute(runUuids: string[], experimentIds: string[]) {
    const queryString = `?runs=${JSON.stringify(runUuids)}&experiments=${JSON.stringify(experimentIds)}`;
    return `${generatePath(RoutePaths.compareRuns)}${queryString}`;
  }

  static get compareRunPageRoute() {
    return RoutePaths.compareRuns;
  }
  static get compareExperimentsPageRoute() {
    return RoutePaths.compareExperiments;
  }
  static getCompareExperimentsPageRoute(experimentIds: string[]) {
    const queryString = `?experiments=${JSON.stringify(experimentIds.slice().sort())}`;
    const path = generatePath(RoutePaths.compareExperimentsSearch, { searchString: 's' });
    return `${path}${queryString}`;
  }
}

export default Routes;
