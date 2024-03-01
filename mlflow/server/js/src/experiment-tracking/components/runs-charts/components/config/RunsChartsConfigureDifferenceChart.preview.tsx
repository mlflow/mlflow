import { FormattedMessage } from 'react-intl';
import {
  DifferenceCardConfigCompareGroup,
  RunsChartsCardConfig,
  RunsChartsDifferenceCardConfig,
} from '../../runs-charts.types';
import { RunsChartsRunData } from '../RunsCharts.common';
import { DifferenceViewPlot } from '../charts/DifferenceViewPlot';

export const RunsChartsConfigureDifferenceChartPreview = ({
  previewData,
  groupBy,
  cardConfig,
  setCardConfig,
}: {
  previewData: RunsChartsRunData[];
  groupBy: string;
  cardConfig: RunsChartsDifferenceCardConfig;
  setCardConfig: (setter: (current: RunsChartsCardConfig) => RunsChartsDifferenceCardConfig) => void;
}) => {
  return (
    <div css={{ width: '100%', overflow: 'auto hidden', height: '100%' }}>
      <DifferenceViewPlot
        previewData={previewData}
        groupBy={groupBy}
        cardConfig={cardConfig}
        setCardConfig={setCardConfig}
      />
    </div>
  );
};
