import { barChartCardDefaultMargin } from '../cards/RunsCompareBarChartCard';
import { CompareChartRunData } from '../charts/CompareRunsCharts.common';
import { CompareRunsMetricsBarPlot } from '../charts/CompareRunsMetricsBarPlot';
import { useCompareRunsTooltip } from '../hooks/useCompareRunsTooltip';
import { RunsCompareBarCardConfig } from '../runs-compare.types';

export const RunsCompareConfigureBarChartPreview = ({
  previewData,
  cardConfig,
}: {
  previewData: CompareChartRunData[];
  cardConfig: RunsCompareBarCardConfig;
}) => {
  const { resetTooltip, setTooltip } = useCompareRunsTooltip(cardConfig);

  return (
    <CompareRunsMetricsBarPlot
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
