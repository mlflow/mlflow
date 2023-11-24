import { RunsChartsRunData } from '../../runs-charts/components/RunsCharts.common';
import { RunsScatterPlot } from '../../runs-charts/components/RunsScatterPlot';
import { useRunsChartsTooltip } from '../../runs-charts/hooks/useRunsChartsTooltip';
import { RunsCompareScatterCardConfig } from '../runs-compare.types';

export const RunsCompareConfigureScatterChartPreview = ({
  previewData,
  cardConfig,
}: {
  previewData: RunsChartsRunData[];
  cardConfig: RunsCompareScatterCardConfig;
}) => {
  const { resetTooltip, setTooltip } = useRunsChartsTooltip(cardConfig);

  return (
    <RunsScatterPlot
      xAxis={cardConfig.xaxis}
      yAxis={cardConfig.yaxis}
      runsData={previewData}
      onHover={setTooltip}
      onUnhover={resetTooltip}
      useDefaultHoverBox={false}
    />
  );
};
