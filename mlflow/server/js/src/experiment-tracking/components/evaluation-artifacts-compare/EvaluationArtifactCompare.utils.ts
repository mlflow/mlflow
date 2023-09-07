import { shouldEnableExperimentDatasetTracking } from '../../../common/utils/FeatureUtils';

export const EVALUATION_ARTIFACTS_TEXT_COLUMN_WIDTH = 240;
export const EVALUATION_ARTIFACTS_RUN_NAME_HEIGHT = 40;
export const EVALUATION_ARTIFACTS_TABLE_ROW_HEIGHT = 190;

const EVALUATION_ARTIFACTS_TABLE_HEADER_HEIGHT_WITH_DATASETS = 110;
const EVALUATION_ARTIFACTS_TABLE_HEADER_HEIGHT_NO_DATASETS = 75;

export const getEvaluationArtifactsTableHeaderHeight = () => {
  if (shouldEnableExperimentDatasetTracking()) {
    return EVALUATION_ARTIFACTS_TABLE_HEADER_HEIGHT_WITH_DATASETS;
  }
  return EVALUATION_ARTIFACTS_TABLE_HEADER_HEIGHT_NO_DATASETS;
};
