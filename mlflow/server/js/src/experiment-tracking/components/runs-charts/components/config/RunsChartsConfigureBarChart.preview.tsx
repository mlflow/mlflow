import { barChartCardDefaultMargin } from '../cards/RunsChartsBarChartCard';
import { RunsChartsRunData } from '../RunsCharts.common';
import { RunsMetricsBarPlot } from '../RunsMetricsBarPlot';
import { useRunsChartsTooltip } from '../../hooks/useRunsChartsTooltip';
import { RunsChartsBarCardConfig } from '../../runs-charts.types';

export const RunsChartsConfigureBarChartPreview = ({
  previewData,
  cardConfig,
}: {
  previewData: RunsChartsRunData[];
  cardConfig: RunsChartsBarCardConfig;
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
