export const shouldEnableRunEvaluationReviewUIWriteFeatures = () => {
  return false;
};

export const shouldEnableTagGrouping = () => {
  return true;
};

export const shouldEnableUnifiedEvalTab = () => {
  return false;
};

/**
 * Page size for MLflow traces 3.0 search api used in eval tab
 */
export const getMlflowTracesSearchPageSize = () => {
  // OSS backend limit is 500
  return 500;
};

/**
 * Total number of traces that will be fetched via mlflow traces 3.0 search api in eval tab
 */
export const getEvalTabTotalTracesLimit = () => {
  return 1000;
};

/**
 * Determines if traces V4 API should be used to fetch traces
 */
export const shouldUseTracesV4API = () => {
  return false;
};

/**
 * Determines if the long-running traces search API should be used.
 * This is only applicable when V4 APIs are enabled.
 * When enabled, trace searches will use an async polling pattern instead of synchronous requests.
 */
export const shouldUseLongRunningTracesAPI = () => {
  return false;
};

export const shouldEnableSessionGrouping = () => {
  return true;
};

/**
 * Determines if the traces table should use infinite paginated queries
 * instead of eagerly fetching all pages in a single query.
 */
export const shouldUseInfinitePaginatedTraces = () => {
  return false;
};

/**
 * Gates use of the POST /mlflow/traces/sessions/search endpoint inside the
 * infinite-pagination path when the traces table is in "group by sessions"
 * mode. REQUIRES shouldUseInfinitePaginatedTraces() to also be true — the
 * session-search endpoint only fits the infinite-pagination model. When
 * disabled, the client pages traces and groups client-side (legacy behavior).
 */
export const shouldUseSessionsSearchAPI = () => {
  return false;
};
