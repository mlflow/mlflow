import { RunsGroupByConfig } from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/utils/experimentPage.group-row-utils';
import { RunsChartsCardConfig, RunsChartsImageCardConfig } from '../../runs-charts.types';
import { RunsChartsRunData } from '../RunsCharts.common';
import { ImageGridPlot } from '../charts/ImageGridPlot';

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
  return (
    <div css={{ width: '100%', overflow: 'auto hidden' }}>
      <ImageGridPlot
        previewData={previewData}
        cardConfig={cardConfig}
        setCardConfig={setCardConfig}
        containerWidth={500}
        groupBy={groupBy}
      />
    </div>
  );
};
