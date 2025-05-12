/**
 * This file aggregates utility functions for enabling features configured by flags.
 * In the OSS version, you can override them in local development by manually changing the return values.
 */
/**
 * Disable legacy charts on "compare runs" page in favor of a new chart view
 */
export const shouldDisableLegacyRunCompareCharts = () => false;

export const shouldEnableExperimentPageAutoRefresh = () => true;

export const shouldEnableRunDetailsPageAutoRefresh = () => true;
/**
 * UI feature preview: enables prompt lab
 */
export const shouldEnablePromptLab = () => true;

export const shouldUseExperimentPageChartViewAsDefault = () => false;

/**
 * Difference view charts feature
 */
export const shouldEnableDifferenceViewCharts = () => true;

/**
 * Keeping original lines when smoothing.
 */
export const shouldEnableChartsOriginalLinesWhenSmoothing = () => false;
/**
 * Enable to use larger color selection (palette of 400 colors) and murmur hashing of
 * run UUIDs for color assignment
 */
export const shouldEnableLargerColorSelection = () => false;
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

export const shouldEnableTracingUI = () => true;
export const shouldEnableRunDetailsPageTracesTab = () => true;
export const shouldUseCompressedExperimentViewSharedState = () => true;
export const shouldEnableUnifiedChartDataTraceHighlight = () => true;
export const shouldUseRegexpBasedAutoRunsSearchFilter = () => true;
export const shouldUseRunRowsVisibilityMap = () => true;
export const isUnstableNestedComponentsMigrated = () => true;
export const shouldUsePredefinedErrorsInExperimentTracking = () => true;

/**
 * Determines if logged models UI (part of model-centric IA shift) is enabled
 */
export const isExperimentLoggedModelsUIEnabled = () => true;
export const isLoggedModelsFilteringAndSortingEnabled = () => true;
export const isRunPageLoggedModelsTableEnabled = () => isExperimentLoggedModelsUIEnabled();

/**
 * Determines if evaluation results online monitoring UI is enabled
 */
export const isExperimentEvalResultsMonitoringUIEnabled = () => false;
export const shouldUseRenamedUnifiedTracesTab = () => true;

/**
 * Flags enabling fetching data via GraphQL for particular views:
 */
export const shouldEnableGraphQLRunDetailsPage = () => false;
export const shouldEnableGraphQLSampledMetrics = () => false;
export const shouldEnableGraphQLModelVersionsForRunDetails = () => false;
export const shouldRerunExperimentUISeeding = () => false;

/**
 * A flag determining if we should display the new models UI.
 */
export const shouldShowModelsNextUI = () => {
  return true;
};
