import { RunsChartsRunData } from '../../runs-charts/components/RunsCharts.common';
import { RunsContourPlot } from '../../runs-charts/components/RunsContourPlot';
import { useRunsChartsTooltip } from '../../runs-charts/hooks/useRunsChartsTooltip';
import { RunsCompareContourCardConfig } from '../runs-compare.types';

export const RunsCompareConfigureContourChartPreview = ({
  previewData,
  cardConfig,
}: {
  previewData: RunsChartsRunData[];
  cardConfig: RunsCompareContourCardConfig;
}) => {
  const { resetTooltip, setTooltip } = useRunsChartsTooltip(cardConfig);

  return (
    <RunsContourPlot
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
