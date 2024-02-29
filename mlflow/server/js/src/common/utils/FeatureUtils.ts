/**
 * This file aggregates utility functions for enabling features configured by flags.
 * In the OSS version, you can override them in local development by manually changing the return values.
 */
/**
 * Disable legacy charts on "compare runs" page in favor of a new chart view
 */
export const shouldDisableLegacyRunCompareCharts = () => false;
/**
 * UI feature preview: displays data lineage (datasets used) in experiment runs
 */
export const shouldEnableExperimentDatasetTracking = () => true;
/**
 * UI feature preview: enables artifact-based ML experiment output data analysis, used for evaluating LLM prediction data
 */
export const shouldEnableArtifactBasedEvaluation = () => true;

/**
 * Enables features related to deep learning: Phase 1.
 * Includes system metrics and sampled metrics retrieval.
 */
export const shouldEnableDeepLearningUI = () => true;

/**
 * Enables features related to deep learning: Phase 2.
 * Includes run grouping and metric chart grouping.
 */
export const shouldEnableDeepLearningUIPhase2 = () => true;

/**
 * Enables features related to deep learning: Phase 3
 * Includes improved runs visibility controls, reworked hover tooltip and improved handling of charts on run details page.
 */
export const shouldEnableDeepLearningUIPhase3 = () => false;

export const shouldUseUnifiedRunCharts = () => false;
/**
 * UI feature preview: enables prompt lab
 */
export const shouldEnablePromptLab = () => true;

export const shouldUsePathRouting = () => false;
export const shouldEnableShareExperimentViewByTags = () => true;

/**
 * A flag determining if we should display the new models UI.
 */
export const shouldShowModelsNextUI = () => {
  return true;
};

// Determines if a new run rows visibility model in the experiment runs table should be used.
export const shouldUseNewRunRowsVisibilityModel = () =>
  shouldEnableDeepLearningUIPhase3() && shouldEnableShareExperimentViewByTags();

// Determines if experiment run row grouping should be enabled.
export const shouldEnableRunGrouping = () =>
  shouldEnableDeepLearningUIPhase2() && shouldEnableShareExperimentViewByTags();

// Determines if metric charts grouping/sections should be enabled.
export const shouldEnableMetricChartsGrouping = () =>
  shouldEnableDeepLearningUIPhase2() && shouldEnableShareExperimentViewByTags();

// Determines if structured table view for logged table artifacts should be enabled.
export const shouldEnableLoggedArtifactTableView = () => shouldEnableDeepLearningUIPhase2();

// Determines if the compact header should be enabled on the experiment page.
export const shouldEnableExperimentPageCompactHeader = () => shouldEnableDeepLearningUIPhase2();
