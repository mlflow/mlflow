/**
 * This file aggregates utility functions for enabling features configured by flags.
 * In the OSS version, you can override them in local development by manually changing the return values.
 */
/**
 * Disable legacy charts on "compare runs" page in favor of a new chart view
 */
export const shouldDisableLegacyRunCompareCharts = () => false;

/**
 * Enables features related to deep learning: Phase 3
 * Includes improved runs visibility controls, reworked hover tooltip and improved handling of charts on run details page.
 */
export const shouldEnableDeepLearningUIPhase3 = () => true;

export const shouldUseUnifiedRunCharts = () => false;

export const shouldEnableExperimentPageAutoRefresh = () => false;

export const shouldEnableRunDetailsPageAutoRefresh = () => false;
/**
 * UI feature preview: enables prompt lab
 */
export const shouldEnablePromptLab = () => true;

export const shouldUseExperimentPageChartViewAsDefault = () => false;

/**
 * Difference view charts feature
 */
export const shouldEnableDifferenceViewCharts = () => false;

/**
 * Image grid charts feature
 */
export const shouldEnableImageGridCharts = () => false;
/**
 * Log table images feature
 */
export const shouldEnableLogTableImages = () => false;
/**
 * Manual range controls feature
 */
export const shouldEnableManualRangeControls = () => false;
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
 * Support tagging multiple runs
 */
export const shouldEnableTaggingMultipleRuns = () => false;
/**
 * Should enable toggling aggregation of individual runs in run groups
 */
export const shouldEnableToggleIndividualRunsInGroups = () => false;

/**
 * Enables toggle controls for hiding charts with no data
 */
export const shouldEnableHidingChartsWithNoData = () => false;

/**
 * Enables draggable and resizable charts grid
 */
export const shouldEnableDraggableChartsGridLayout = () => false;

/**
 * Update relative time axis to use date
 */
export const shouldEnableRelativeTimeDateAxis = () => false;
/**
 * Should enable new difference view charts
 */
export const shouldEnableNewDifferenceViewCharts = () => false;

export const shouldEnableTracingUI = () => true;
export const shouldEnableRunDetailsPageTracesTab = () => false;
export const shouldUseCompressedExperimentViewSharedState = () => true;
export const shouldEnableUnifiedChartDataTraceHighlight = () => true;
export const shouldDeferLineChartRendering = () => true;
export const shouldEnableGlobalLineChartConfig = () => false;
export const shouldUseRegexpBasedChartFiltering = () => false;
export const shouldUseRegexpBasedAutoRunsSearchFilter = () => false;

/**
 * Flags enabling fetching data via GraphQL for particular views:
 */
export const shouldEnableGraphQLRunDetailsPage = () => false;

/**
 * A flag determining if we should display the new models UI.
 */
export const shouldShowModelsNextUI = () => {
  return true;
};

// Determines if a new run rows visibility model in the experiment runs table should be used.
export const shouldUseNewRunRowsVisibilityModel = () => shouldEnableDeepLearningUIPhase3();

// Determines if improved sort selector should be enabled on the experiment page.
export const shouldUseNewExperimentPageSortSelector = () => shouldEnableDeepLearningUIPhase3();
