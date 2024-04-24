import { useDesignSystemTheme } from '@databricks/design-system';
import { RunsChartsTooltipMode, useRunsChartsTooltip } from '../runs-charts/hooks/useRunsChartsTooltip';
import { useMemo } from 'react';
import { RunsMetricsLinePlot, RunsMetricsLinePlotProps } from '../runs-charts/components/RunsMetricsLinePlot';
import { isSystemMetricKey } from '../../utils/MetricsUtils';
import type { MetricEntity, RunInfoEntity } from '../../types';
import { shouldEnableDeepLearningUIPhase3 } from '../../../common/utils/FeatureUtils';
import { RunsChartsLineChartXAxisType } from '../runs-charts/components/RunsCharts.common';

export interface RunViewMetricHistoryChartProps extends Pick<RunsMetricsLinePlotProps, 'xRange' | 'yRange'> {
  metricKey: string;
  runInfo: RunInfoEntity;
  metricsHistory: MetricEntity[];
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
  xRange,
  yRange,
}: RunViewMetricHistoryChartProps) => {
  const { theme } = useDesignSystemTheme();

  const usingMultipleRunsHoverTooltip = shouldEnableDeepLearningUIPhase3();

  const { setTooltip, resetTooltip } = useRunsChartsTooltip(
    { metricKey },
    usingMultipleRunsHoverTooltip ? RunsChartsTooltipMode.MultipleTracesWithScanline : RunsChartsTooltipMode.Simple,
  );

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
      xAxisKey={isSystemMetricKey(metricKey) ? RunsChartsLineChartXAxisType.TIME : RunsChartsLineChartXAxisType.STEP}
      selectedXAxisMetricKey=""
      onHover={setTooltip}
      onUnhover={resetTooltip}
      useDefaultHoverBox={false}
      lineSmoothness={0}
      xRange={xRange}
      yRange={yRange}
      onUpdate={onUpdate}
    />
  );
};
