import type { RunsChartsImageCardConfig, RunsChartsCardConfig } from '../../runs-charts.types';
import type { RunsChartsRunData } from '../RunsCharts.common';
import { ImageGridSingleKeyPlot } from './ImageGridSingleKeyPlot';
import { ImageGridMultipleKeyPlot } from './ImageGridMultipleKeyPlot';
import {
  LOG_IMAGE_TAG_INDICATOR,
  NUM_RUNS_TO_SUPPORT_FOR_LOG_IMAGE,
} from '@mlflow/mlflow/src/experiment-tracking/constants';
import type { RunsGroupByConfig } from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/utils/experimentPage.group-row-utils';
import { Empty } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

export const ImageGridPlot = ({
  previewData,
  cardConfig,
  groupBy,
  setCardConfig,
}: {
  previewData: RunsChartsRunData[];
  cardConfig: RunsChartsImageCardConfig;
  groupBy: RunsGroupByConfig | null;
  setCardConfig?: (setter: (current: RunsChartsCardConfig) => RunsChartsImageCardConfig) => void;
}) => {
  const containsLoggedImages = previewData.some((run: RunsChartsRunData) => Boolean(run.tags[LOG_IMAGE_TAG_INDICATOR]));

  const filteredPreviewData = previewData
    .filter((run: RunsChartsRunData) => {
      return run.tags[LOG_IMAGE_TAG_INDICATOR];
    })
    .slice(-NUM_RUNS_TO_SUPPORT_FOR_LOG_IMAGE);

  if (!containsLoggedImages) {
    return (
      <div css={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <Empty
          title={
            <FormattedMessage
              defaultMessage="No images found"
              description="Title for the empty state when no images are found in the currently visible runs"
            />
          }
          description={
            <FormattedMessage
              defaultMessage="No logged images found in the currently visible runs"
              description="Description for the empty state when no images are found in the currently visible runs"
            />
          }
        />
      </div>
    );
  }

  if (cardConfig.imageKeys.length === 1) {
    return <ImageGridSingleKeyPlot previewData={filteredPreviewData} cardConfig={cardConfig} />;
  } else if (cardConfig.imageKeys.length > 1) {
    return <ImageGridMultipleKeyPlot previewData={filteredPreviewData} cardConfig={cardConfig} />;
  }
  return null;
};
