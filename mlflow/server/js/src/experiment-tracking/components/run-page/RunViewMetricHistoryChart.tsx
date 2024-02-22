import { useDesignSystemTheme } from '@databricks/design-system';
import { useRunsChartsTooltip } from '../runs-charts/hooks/useRunsChartsTooltip';
import { useMemo } from 'react';
import { RunsMetricsLinePlot, RunsMetricsLinePlotProps } from '../runs-charts/components/RunsMetricsLinePlot';
import { isSystemMetricKey } from '../../utils/MetricsUtils';
import type { MetricEntity, RunInfoEntity } from '../../types';

export interface RunViewMetricHistoryChartProps {
  metricKey: string;
  runInfo: RunInfoEntity;
  metricsHistory: MetricEntity[];
  range: RunsMetricsLinePlotProps['range'];
  onUpdate: RunsMetricsLinePlotProps['onUpdate'];
}

/**
 * Chart variant displaying metric history using line plot
 */
export const RunViewMetricHistoryChart = ({
  runInfo,
  metricKey,
  metricsHistory,
  onUpdate,
  range,
}: RunViewMetricHistoryChartProps) => {
  const { theme } = useDesignSystemTheme();
  const { setTooltip, resetTooltip } = useRunsChartsTooltip({ metricKey });

  // Prepare a single trace for the line chart
  const chartData = useMemo(
    () => [
      {
        uuid: runInfo.run_uuid,
        displayName: runInfo.run_name,
        runInfo,
        metricsHistory: { [metricKey]: metricsHistory },
        color: theme.colors.primary,
      },
    ],
    [runInfo, metricsHistory, metricKey, theme],
  );

  return (
    <RunsMetricsLinePlot
      metricKey={metricKey}
      runsData={chartData}
      xAxisKey={isSystemMetricKey(metricKey) ? 'time' : 'step'}
      onHover={setTooltip}
      onUnhover={resetTooltip}
      useDefaultHoverBox={false}
      lineSmoothness={0}
      range={range}
      onUpdate={onUpdate}
      lockXAxisZoom
    />
  );
};
