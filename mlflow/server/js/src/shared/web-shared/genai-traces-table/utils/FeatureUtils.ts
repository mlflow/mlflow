export const shouldEnableRunEvaluationReviewUIWriteFeatures = () => {
  return false;
};

export const shouldEnableTagGrouping = () => {
  return true;
};

export const shouldEnableUnifiedEvalTab = () => {
  return false;
};

export const shouldUseRunIdFilterInSearchTraces = () => {
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
