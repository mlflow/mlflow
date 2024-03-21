import { RunsChartsRunData } from '../RunsCharts.common';
import { RunsContourPlot } from '../RunsContourPlot';
import { useRunsChartsTooltip } from '../../hooks/useRunsChartsTooltip';
import { RunsChartsContourCardConfig } from '../../runs-charts.types';

export const RunsChartsConfigureContourChartPreview = ({
  previewData,
  cardConfig,
}: {
  previewData: RunsChartsRunData[];
  cardConfig: RunsChartsContourCardConfig;
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
