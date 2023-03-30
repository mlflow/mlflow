import { CompareChartRunData } from '../charts/CompareRunsCharts.common';
import { CompareRunsScatterPlot } from '../charts/CompareRunsScatterPlot';
import { useCompareRunsTooltip } from '../hooks/useCompareRunsTooltip';
import { RunsCompareScatterCardConfig } from '../runs-compare.types';

export const RunsCompareConfigureScatterChartPreview = ({
  previewData,
  cardConfig,
}: {
  previewData: CompareChartRunData[];
  cardConfig: RunsCompareScatterCardConfig;
}) => {
  const { resetTooltip, setTooltip } = useCompareRunsTooltip(cardConfig);

  return (
    <CompareRunsScatterPlot
      xAxis={cardConfig.xaxis}
      yAxis={cardConfig.yaxis}
      runsData={previewData}
      onHover={setTooltip}
      onUnhover={resetTooltip}
      useDefaultHoverBox={false}
    />
  );
};
