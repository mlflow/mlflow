import { barChartCardDefaultMargin } from '../cards/RunsCompareBarChartCard';
import { RunsChartsRunData } from '../../runs-charts/components/RunsCharts.common';
import { RunsMetricsBarPlot } from '../../runs-charts/components/RunsMetricsBarPlot';
import { useRunsChartsTooltip } from '../../runs-charts/hooks/useRunsChartsTooltip';
import { RunsCompareBarCardConfig } from '../runs-compare.types';

export const RunsCompareConfigureBarChartPreview = ({
  previewData,
  cardConfig,
}: {
  previewData: RunsChartsRunData[];
  cardConfig: RunsCompareBarCardConfig;
}) => {
  const { resetTooltip, setTooltip } = useRunsChartsTooltip(cardConfig);

  return (
    <RunsMetricsBarPlot
      useDefaultHoverBox={false}
      displayRunNames={false}
      displayMetricKey={false}
      metricKey={cardConfig.metricKey}
      runsData={previewData}
      margin={barChartCardDefaultMargin}
      onHover={setTooltip}
      onUnhover={resetTooltip}
    />
  );
};
