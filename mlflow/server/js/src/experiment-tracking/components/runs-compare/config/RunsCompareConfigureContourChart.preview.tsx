import { CompareChartRunData } from '../charts/CompareRunsCharts.common';
import { CompareRunsContourPlot } from '../charts/CompareRunsContourPlot';
import { useCompareRunsTooltip } from '../hooks/useCompareRunsTooltip';
import { RunsCompareContourCardConfig } from '../runs-compare.types';

export const RunsCompareConfigureContourChartPreview = ({
  previewData,
  cardConfig,
}: {
  previewData: CompareChartRunData[];
  cardConfig: RunsCompareContourCardConfig;
}) => {
  const { resetTooltip, setTooltip } = useCompareRunsTooltip(cardConfig);

  return (
    <CompareRunsContourPlot
      xAxis={cardConfig.xaxis}
      yAxis={cardConfig.yaxis}
      zAxis={cardConfig.zaxis}
      runsData={previewData}
      onHover={setTooltip}
      onUnhover={resetTooltip}
      useDefaultHoverBox={false}
    />
  );
};
