import { createMLflowRoutePath, generatePath } from '../common/utils/RoutingUtils';
import type { ExperimentPageTabName } from './constants';

/**
 * Page identifiers for MLflow experiment tracking pages.
 * Keys should correspond to route paths.
 */
export enum PageId {
  home = 'mlflow.home',
  promptsPage = 'mlflow.prompts',
  promptDetailsPage = 'mlflow.prompts.details',
  experimentPageTabbed = 'mlflow.experiment.details.tab',
  experimentLoggedModelDetailsPageTab = 'mlflow.logged-model.details.tab',
  experimentLoggedModelDetailsPage = 'mlflow.logged-model.details',
  experimentPage = 'mlflow.experiment.details',
  // Child routes for experiment page:
  experimentPageTabRuns = 'mlflow.experiment.tab.runs',
  experimentPageTabModels = 'mlflow.experiment.tab.models',
  experimentPageTabTraces = 'mlflow.experiment.tab.traces',
  // Child routes for experiment page - end
  experimentPageSearch = 'mlflow.experiment.details.search',
  compareExperimentsSearch = 'mlflow.experiment.compare',
  runPageWithTab = 'mlflow.experiment.run.details',
  runPageDirect = 'mlflow.experiment.run.details.direct',
  compareRuns = 'mlflow.experiment.run.compare',
  metricPage = 'mlflow.metric.details',
  experimentPrompt = 'mlflow.experiment.prompt',
}

// Route path definitions (used in defining route elements)
// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
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
  // Child routes for experiment page:
  static get experimentPageTabRuns() {
    return createMLflowRoutePath('/experiments/:experimentId/runs');
  }
  static get experimentPageTabTraces() {
    return createMLflowRoutePath('/experiments/:experimentId/traces');
  }
  static get experimentPageTabModels() {
    return createMLflowRoutePath('/experiments/:experimentId/models');
  }
  // Child routes for experiment page - end
  static get experimentLoggedModelDetailsPageTab() {
    return createMLflowRoutePath('/experiments/:experimentId/models/:loggedModelId/:tabName');
  }
  static get experimentLoggedModelDetailsPage() {
    return createMLflowRoutePath('/experiments/:experimentId/models/:loggedModelId');
  }
  static get experimentPageTabbed() {
    return createMLflowRoutePath('/experiments/:experimentId/:tabName');
  }
  static get experimentPageSearch() {
    return createMLflowRoutePath('/experiments/:experimentId/s');
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
  static get experimentPromptsList() {
    return createMLflowRoutePath('/experiments/:experimentId/prompts');
  }
  static get experimentPrompt() {
    return createMLflowRoutePath('/experiments/:experimentId/prompts/:promptName');
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
  /**
   * Route paths for prompts management.
   * Featured exclusively in open source MLflow.
   */
  static get promptsPage() {
    return createMLflowRoutePath('/prompts');
  }
  static get promptDetailsPage() {
    return createMLflowRoutePath('/prompts/:promptName');
  }
}

// Concrete routes and functions for generating parametrized paths
// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
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

  static getExperimentPageTracesTabRoute(experimentId: string) {
    return `${Routes.getExperimentPageRoute(experimentId)}/traces`;
  }

  static getExperimentPageTabRoute(experimentId: string, tabName: ExperimentPageTabName) {
    return generatePath(RoutePaths.experimentPageTabbed, { experimentId, tabName });
  }

  static getExperimentLoggedModelDetailsPage(experimentId: string, loggedModelId: string) {
    return generatePath(RoutePaths.experimentLoggedModelDetailsPage, { experimentId, loggedModelId });
  }

  static getExperimentLoggedModelDetailsPageRoute(experimentId: string, loggedModelId: string, tabName?: string) {
    if (tabName) {
      return generatePath(RoutePaths.experimentLoggedModelDetailsPageTab, { experimentId, loggedModelId, tabName });
    }
    return generatePath(RoutePaths.experimentLoggedModelDetailsPage, { experimentId, loggedModelId });
  }

  static searchRunsByUser(experimentId: string, userId: string) {
    const path = generatePath(RoutePaths.experimentPage, { experimentId });
    const filterString = `attributes.user_id = '${userId}'`;
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

  static getDirectRunPageRoute(runUuid: string) {
    return generatePath(RoutePaths.runPageDirect, { runUuid });
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
    selectedXAxis: 'wall' | 'step' | 'relative' = 'step',
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

  /**
   * Routes for prompts management.
   * Featured exclusively in open source MLflow.
   */
  static get promptsPageRoute() {
    return RoutePaths.promptsPage;
  }

  static getPromptDetailsPageRoute(promptName: string) {
    return generatePath(RoutePaths.promptDetailsPage, { promptName });
  }
}

export default Routes;
