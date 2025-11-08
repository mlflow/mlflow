import type { RunsGroupByConfig } from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/utils/experimentPage.group-row-utils';
import type { RunsChartsCardConfig, RunsChartsImageCardConfig } from '../../runs-charts.types';
import type { RunsChartsRunData } from '../RunsCharts.common';
import { ImageGridPlot } from '../charts/ImageGridPlot';
import { FormattedMessage } from 'react-intl';
import { Empty } from '@databricks/design-system';
import { LOG_IMAGE_TAG_INDICATOR } from '../../../../constants';

export const RunsChartsConfigureImageChartPreview = ({
  previewData,
  cardConfig,
  setCardConfig,
  groupBy,
}: {
  previewData: RunsChartsRunData[];
  cardConfig: RunsChartsImageCardConfig;
  setCardConfig: (setter: (current: RunsChartsCardConfig) => RunsChartsImageCardConfig) => void;
  groupBy: RunsGroupByConfig | null;
}) => {
  const containsLoggedImages = previewData.some((run: RunsChartsRunData) => Boolean(run.tags[LOG_IMAGE_TAG_INDICATOR]));

  if (containsLoggedImages && cardConfig?.imageKeys?.length === 0) {
    return (
      <Empty
        title={
          <FormattedMessage
            defaultMessage="No images configured for preview"
            description="Title for the empty state when user did not configure any images for preview yet"
          />
        }
        description={
          <FormattedMessage
            defaultMessage="Please use controls on the left to select images to be compared"
            description="Description for the empty state when user did not configure any images for preview yet"
          />
        }
      />
    );
  }

  const chartBody = (
    <ImageGridPlot previewData={previewData} cardConfig={cardConfig} setCardConfig={setCardConfig} groupBy={groupBy} />
  );

  const cardBodyToRender = chartBody;

  return <div css={{ width: '100%', overflow: 'auto hidden' }}>{cardBodyToRender}</div>;
};
