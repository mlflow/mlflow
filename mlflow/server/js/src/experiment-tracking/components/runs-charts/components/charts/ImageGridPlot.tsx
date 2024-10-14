import { useDispatch } from 'react-redux';
import { RunsChartsImageCardConfig, RunsChartsCardConfig } from '../../runs-charts.types';
import { RunsChartsRunData } from '../RunsCharts.common';
import { ThunkDispatch } from '@mlflow/mlflow/src/redux-types';
import { useEffect } from 'react';
import { shouldEnableImageGridCharts } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { ImageGridSingleKeyPlot } from './ImageGridSingleKeyPlot';
import { ImageGridMultipleKeyPlot } from './ImageGridMultipleKeyPlot';
import {
  DEFAULT_IMAGE_GRID_CHART_NAME,
  LOG_IMAGE_TAG_INDICATOR,
  NUM_RUNS_TO_SUPPORT_FOR_LOG_IMAGE,
} from '@mlflow/mlflow/src/experiment-tracking/constants';
import { RunsGroupByConfig } from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/utils/experimentPage.group-row-utils';

export const ImageGridPlot = ({
  previewData,
  cardConfig,
  groupBy,
  setCardConfig,
  containerWidth,
}: {
  previewData: RunsChartsRunData[];
  cardConfig: RunsChartsImageCardConfig;
  groupBy: RunsGroupByConfig | null;
  setCardConfig?: (setter: (current: RunsChartsCardConfig) => RunsChartsImageCardConfig) => void;
  containerWidth: number;
}) => {
  const filteredPreviewData = previewData
    .filter((run: RunsChartsRunData) => {
      return run.tags[LOG_IMAGE_TAG_INDICATOR];
    })
    .slice(-NUM_RUNS_TO_SUPPORT_FOR_LOG_IMAGE);

  if (cardConfig.imageKeys.length === 1) {
    return <ImageGridSingleKeyPlot previewData={filteredPreviewData} cardConfig={cardConfig} width={containerWidth} />;
  } else if (cardConfig.imageKeys.length > 1) {
    return (
      <ImageGridMultipleKeyPlot previewData={filteredPreviewData} cardConfig={cardConfig} width={containerWidth} />
    );
  }
  return null;
};
