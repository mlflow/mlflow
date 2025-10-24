/**
 * This file aggregates utility functions for enabling features configured by flags.
 * In the OSS version, you can override them in local development by manually changing the return values.
 */

export const shouldEnableRunDetailsPageAutoRefresh = () => true;
/**
 * UI feature preview: enables prompt lab
 */
export const shouldEnablePromptLab = () => true;

/**
 * Enable chart expressions feature
 */
export const shouldEnableChartExpressions = () => false;
/**
 * Should enable toggling aggregation of individual runs in run groups
 */
export const shouldEnableToggleIndividualRunsInGroups = () => false;

/**
 * Update relative time axis to use date
 */
export const shouldEnableRelativeTimeDateAxis = () => false;
/**
 * Should enable new difference view charts
 */
export const shouldEnableNewDifferenceViewCharts = () => false;
export const shouldEnableDifferenceViewChartsV3 = () => false;
export const shouldEnableMinMaxMetricsOnExperimentPage = () => false;

export const shouldUseCompressedExperimentViewSharedState = () => true;
export const shouldEnableUnifiedChartDataTraceHighlight = () => true;
export const shouldUseRegexpBasedAutoRunsSearchFilter = () => true;
export const shouldUseRunRowsVisibilityMap = () => true;
export const isUnstableNestedComponentsMigrated = () => true;
export const shouldUsePredefinedErrorsInExperimentTracking = () => true;

/**
 * Determines if logged models UI (part of model-centric IA shift) is enabled
 */
export const isLoggedModelsFilteringAndSortingEnabled = () => false;
export const isRunPageLoggedModelsTableEnabled = () => true;

/**
 * Flags enabling fetching data via GraphQL for particular views:
 */
export const shouldEnableGraphQLRunDetailsPage = () => true;
export const shouldEnableGraphQLSampledMetrics = () => false;
export const shouldEnableGraphQLModelVersionsForRunDetails = () => false;
export const shouldRerunExperimentUISeeding = () => false;

/**
 * Determines if experiment kind inference is enabled.
 */
export const shouldEnableExperimentKindInference = () => true;

/**
 * Determines if the new prompts tab on DB platform is enabled.
 */
export const shouldEnablePromptsTabOnDBPlatform = () => false;

export const shouldEnablePromptTags = () => false;

export const shouldUseSharedTaggingUI = () => false;

export const shouldDisableReproduceRunButton = () => false;

export const shouldUnifyLoggedModelsAndRegisteredModels = () => {
  return false;
};

/**
 * Enables use of GetLoggedModels API allowing to get multiple logged models by their IDs.
 */
export const shouldUseGetLoggedModelsBatchAPI = () => {
  return false;
};

/**
 * Uses restructured routes for experiment page: parent+child hierarchy with <Outlet> instead of tab parameter.
 */
export const shouldEnableExperimentPageChildRoutes = () => {
  return false;
};

/**
 * A flag determining if we should display the new models UI.
 */
export const shouldShowModelsNextUI = () => {
  return true;
};

export const shouldEnableTracesV3View = () => {
  return true;
};

export const shouldEnableTraceInsights = () => {
  return false;
};

export const shouldEnableTracesSyncUI = () => {
  return false;
};

/**
 * Total number of traces that will be fetched via mlflow traces 3.0 search api in eval tab
 */
export const getEvalTabTotalTracesLimit = () => {
  return 1000;
};

/**
 * Determines if evaluation results online monitoring UI is enabled
 */
export const isExperimentEvalResultsMonitoringUIEnabled = () => {
  return true;
};

export const shouldUseUnifiedArtifactBrowserForLoggedModels = () => {
  return false;
};

export const shouldUseUnifiedArtifactBrowserForRunDetailsPage = () => {
  return false;
};

export const shouldEnableTagGrouping = () => {
  return true;
};

/**
 * Determines if the assessments pane should be disabled when trace info fetch fails.
 * In OSS, we keep the pane enabled to avoid confusing users (showing stale data is better than nothing).
 * In Databricks, we disable it because playground creates fake traces that can't have assessments.
 */
export const shouldDisableAssessmentsPaneOnFetchFailure = () => {
  return false;
};
