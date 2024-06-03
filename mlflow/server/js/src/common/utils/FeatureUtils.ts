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

export const shouldEnableRunsTableRunNameColumnResize = () => false;
/**
 * Image grid charts feature
 */
export const shouldEnableImageGridCharts = () => false;
/**
 * Log table images feature
 */
export const shouldEnableLogTableImages = () => false;

/**
 * Should enable toggling aggregation of individual runs in run groups
 */
export const shouldEnableToggleIndividualRunsInGroups = () => false;

/**
 * Update relative time axis to use date
 */
export const shouldEnableRelativeTimeDateAxis = () => false;

export const shouldEnableTracingUI = () => true;

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
