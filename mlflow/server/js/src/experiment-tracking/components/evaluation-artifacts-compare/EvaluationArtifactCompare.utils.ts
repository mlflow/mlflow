export const EVALUATION_ARTIFACTS_TEXT_COLUMN_WIDTH = {
  // Default width of "group by" columns
  initialWidthGroupBy: 200,
  // Default width of "compare" (output) columns
  initialWidthOutput: 360,
  maxWidth: 500,
  minWidth: 140,
};
export const EVALUATION_ARTIFACTS_TABLE_ROW_HEIGHT = 190;

export const getEvaluationArtifactsTableHeaderHeight = (isExpanded = false, includePlaceForMetadata = false) => {
  // If there is no metadata displayed at all, prepare
  // 40 px for group header plus 40 px for the run name
  if (!includePlaceForMetadata) {
    return 80;
  }

  // If there's a metadata to be displayed, base the resulting height
  // on the header expansion. Pixel values according to designs.
  return 40 + (isExpanded ? 175 : 62);
};
