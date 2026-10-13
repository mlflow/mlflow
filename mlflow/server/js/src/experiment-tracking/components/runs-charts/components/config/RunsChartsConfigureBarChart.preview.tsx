import { barChartCardDefaultMargin } from '../cards/RunsChartsBarChartCard';
import type { RunsChartsRunData } from '../RunsCharts.common';
import { RunsMetricsBarPlot } from '../RunsMetricsBarPlot';
import { useRunsChartsTooltip } from '../../hooks/useRunsChartsTooltip';
import type { RunsChartsBarCardConfig } from '../../runs-charts.types';

export const RunsChartsConfigureBarChartPreview = ({
  previewData,
  cardConfig,
}: {
  previewData: RunsChartsRunData[];
  cardConfig: RunsChartsBarCardConfig;
}) => {
  const { resetTooltip, setTooltip } = useRunsChartsTooltip(cardConfig);

  const primaryDataKey = cardConfig.dataAccessKey ?? cardConfig.metricKey;
  const selectedMetricKeys = cardConfig.selectedMetricKeys ?? [primaryDataKey];
  const isMultiMetric = selectedMetricKeys.length > 1;

  return (
    <RunsMetricsBarPlot
      useDefaultHoverBox={false}
      displayRunNames={false}
      displayMetricKey={false}
      metricKey={primaryDataKey}
      selectedMetricKeys={isMultiMetric ? selectedMetricKeys : undefined}
      runsData={previewData}
      margin={barChartCardDefaultMargin}
      onHover={setTooltip}
      onUnhover={resetTooltip}
    />
  );
};
