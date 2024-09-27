import { useMemo } from 'react';
import { RunsMetricsBarPlot } from '../runs-charts/components/RunsMetricsBarPlot';
import { useDesignSystemTheme } from '@databricks/design-system';
import { useRunsChartsTooltip } from '../runs-charts/hooks/useRunsChartsTooltip';
import { MetricEntity, RunInfoEntity } from '../../types';
import type { UseGetRunQueryResponseRunInfo } from './hooks/useGetRunQuery';

export interface RunViewSingleMetricChartProps {
  metricKey: string;
  runInfo: RunInfoEntity | UseGetRunQueryResponseRunInfo;
  metricEntry?: MetricEntity;
}

/**
 * Chart variant displaying single (non-history) value using bar plot
 */
export const RunViewMetricSingleValueChart = ({ runInfo, metricKey, metricEntry }: RunViewSingleMetricChartProps) => {
  const { theme } = useDesignSystemTheme();
  const { setTooltip, resetTooltip } = useRunsChartsTooltip({ metricKey });

  // Prepare a single trace for the line chart
  const chartData = useMemo(
    () =>
      metricEntry
        ? [
            {
              uuid: runInfo.runUuid ?? '',
              displayName: runInfo.runName ?? '',
              runInfo,
              metrics: { [metricKey]: metricEntry },
              color: theme.colors.primary,
            },
          ]
        : [],
    [runInfo, metricEntry, metricKey, theme],
  );

  return (
    <RunsMetricsBarPlot
      metricKey={metricKey}
      runsData={chartData}
      height={260}
      onHover={setTooltip}
      onUnhover={resetTooltip}
      useDefaultHoverBox={false}
      displayRunNames={false}
      displayMetricKey={false}
    />
  );
};
