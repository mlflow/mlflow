/**
 * This file aggregates utility functions for enabling features configured by flags.
 * In the OSS version, you can override them in local development by manually changing the return values.
 */

import { getWorkspacesEnabledSync } from '../../experiment-tracking/hooks/useServerInfo';

// Returns the current workspaces enabled state from the cached server features.
// This is synchronous and returns the cached value (false if not yet loaded).
// For React components, prefer using the useWorkspacesEnabled hook instead.
export const shouldEnableWorkspaces = () => {
  return getWorkspacesEnabledSync();
};

/**
 * Enable chart expressions feature
 */
export const shouldEnableChartExpressions = () => false;

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
export const shouldUseRegexpBasedAutoRunsSearchFilter = () => false;
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

/**
 * Feature flag to enable Scorers UI tab in experiment page
 */
export const enableScorersUI = () => {
  return true;
};

/**
 * Determines if the new GenAI experiment creation modal with table prefix onboarding is enabled.
 * When enabled, the observatory shows a create modal with UC storage selection and table prefix,
 * and the inline UC schema selector in the traces toolbar is hidden.
 */
export const shouldEnableGenAIExperimentCreationModal = () => {
  return false;
};

/**
 * Determines if running scorers feature is enabled (ability to run LLM scorers on sample traces)
 */
export const isRunningScorersEnabled = () => {
  return true;
};

/**
 * Determines if evaluating sessions (not just traces) in scorers is enabled.
 * When false, session-level scorers cannot be run on sample traces. They can still be created.
 */
export const isEvaluatingSessionsInScorersEnabled = () => {
  return true;
};

/**
 * Determines if running agentic judges (judges using the {{ trace }} variable) is enabled
 * in the sample scorer output panel.
 */
export const isRunningAgenticJudgesEnabled = () => {
  return true;
};

/**
 * Determines if all scorer templates are supported for running on sample traces.
 * When false, only templates with chat-assessments mapping or editable instructions are supported.
 */
export const isRunningAllScorerTemplatesEnabled = () => {
  return true;
};

/**
 * Determines if the output type selector is shown in the LLM scorer form.
 * When false, the output type defaults to 'default' (no explicit type sent to API).
 */
export const isScorerOutputTypeSelectorEnabled = () => {
  return true;
};

/**
 * Scorer pagination is supported in managed but not oss.
 */
export const shouldPaginateScorers = () => {
  return false;
};

/**
 * Determines if the new prompts tab on DB platform is enabled.
 */
export const shouldEnablePromptsTabOnDBPlatform = () => false;

export const shouldEnablePromptTags = () => false;

export const shouldUseSharedTaggingUI = () => false;

export const shouldDisableReproduceRunButton = () => false;

export const shouldEnablePromptLab = () => {
  return true;
};

export const shouldEnableNodeLevelSystemMetricCharts = () => {
  return false;
};

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
 * A flag determining if we should display the new models UI.
 */
export const shouldShowModelsNextUI = () => {
  return true;
};

export const shouldEnableTracesSyncUI = () => {
  return false;
};

/**
 * Whether to batch multiple token metric queries into a single QueryTraceMetrics call
 * using the metric_names (plural) field. Requires backend support for the new field.
 */
export const shouldEnableBatchedTokenMetricQueries = () => {
  // TODO: enable this when the backend is ready
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

/**
 * Determines if the run metadata are visible on run details page overview.
 */
export const shouldEnableRunDetailsMetadataBoxOnRunDetailsPage = () => {
  return false;
};

/**
 * Determines if the artifacts are visible on run details page overview.
 */
export const shouldEnableArtifactsOnRunDetailsPage = () => {
  return false;
};

/**
 * Whether the Overview tab should be shown for a given experiment.
 *
 * On Databricks, requires the rollout flag. For UC-backed experiments (hasV4Location=true),
 * the overview tab is always shown. For MySQL-backed experiments, a separate
 * enableMysqlExperimentOverview flag must also be enabled.
 * On OSS (after edge stripping), the tab is always enabled.
 *
 * @param hasV4Location — true when the experiment's trace storage is UC-backed.
 *   Sourced from SqlWarehouseContext; undefined when no provider is present (OSS).
 */
export const shouldEnableExperimentOverviewTab = (hasV4Location?: boolean) => {
  return true;
};

/**
 * Determines if the top-level nested sidebar feature is enabled.
 * This enables the workflow type selector and nested navigation items in the main sidebar.
 */
export const shouldEnableWorkflowBasedNavigation = () => {
  return true;
};

/**
 * Enables improved evaluation runs comparison UI with full-page list view,
 * dataset grouping, and streamlined run comparison workflow.
 */
export const shouldEnableImprovedEvalRunsComparison = () => {
  return false;
};

export const isScorerModelSelectionEnabled = () => {
  return true;
};

/**
 * Determines if issue detection feature is enabled in the traces table toolbar.
 */
export const shouldEnableIssueDetection = () => {
  return true;
};

/**
 * Controls visibility of the right panel (issues) on the evaluation runs page.
 * When enabled (true), the right panel is hidden by default and only the evaluation runs table is shown.
 */
export const shouldShowEvalRunsIssuesPanel = () => {
  return true;
};

/**
 * Determines if databricks:/ provider models can be run from the UI.
 * In Databricks, databricks:/ models are gateway-routed and runnable.
 * In OSS (after Copybara strips EDGE), databricks:/ models are not supported.
 */
export const shouldSupportRunningDatabricksProviderJudgesFromUI = () => {
  return false;
};
