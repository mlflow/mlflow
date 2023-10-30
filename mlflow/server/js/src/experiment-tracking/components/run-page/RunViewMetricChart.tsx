import { first } from 'lodash';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { MetricHistoryByName, RunInfoEntity } from '../../types';
import { LegacySkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { useRunsChartsTooltip } from '../runs-charts/hooks/useRunsChartsTooltip';
import { isSystemMetricKey } from '../../utils/MetricsUtils';
import { RunsMetricsLinePlot } from '../runs-charts/components/RunsMetricsLinePlot';
import { RunsMetricsBarPlot } from '../runs-charts/components/RunsMetricsBarPlot';

export interface RunViewMetricChartProps {
  metricKey: string;
  runInfo: RunInfoEntity;
  metricsHistory: MetricHistoryByName;
  isLoading: boolean;
}

/**
 * Chart variant displaying metric history using line plot
 */
const RunViewMetricHistoryChart = ({
  runInfo,
  metricKey,
  metricsHistory,
}: Omit<RunViewMetricChartProps, 'isLoading'>) => {
  const { theme } = useDesignSystemTheme();
  const { setTooltip, resetTooltip } = useRunsChartsTooltip({ metricKey });

  // Prepare a single trace for the line chart
  const chartData = useMemo(
    () => [
      {
        runInfo,
        metricsHistory,
        color: theme.colors.primary,
      },
    ],
    [runInfo, metricsHistory, theme],
  );

  return (
    <RunsMetricsLinePlot
      metricKey={metricKey}
      runsData={chartData}
      height={300}
      xAxisKey={isSystemMetricKey(metricKey) ? 'time' : 'step'}
      onHover={setTooltip}
      onUnhover={resetTooltip}
      useDefaultHoverBox={false}
      lineSmoothness={0}
    />
  );
};

/**
 * Chart variant displaying single (non-history) value using bar plot
 */
const RunViewMetricSingleValueChart = ({
  runInfo,
  metricKey,
  metricsHistory,
}: Omit<RunViewMetricChartProps, 'isLoading'>) => {
  const { theme } = useDesignSystemTheme();
  const { setTooltip, resetTooltip } = useRunsChartsTooltip({ metricKey });

  const firstMetricEntry = first(metricsHistory[metricKey]);

  // Prepare a single trace for the line chart
  const chartData = useMemo(
    () =>
      firstMetricEntry
        ? [
            {
              runInfo,
              metrics: { [metricKey]: firstMetricEntry },
              color: theme.colors.primary,
            },
          ]
        : [],
    [runInfo, firstMetricEntry, metricKey, theme],
  );

  return (
    <RunsMetricsBarPlot
      metricKey={metricKey}
      runsData={chartData}
      height={300}
      onHover={setTooltip}
      onUnhover={resetTooltip}
      useDefaultHoverBox={false}
      displayRunNames={false}
      displayMetricKey={false}
      barLabelTextPosition='inside'
    />
  );
};

/**
 * A single chart component displayed in run view metric charts tab.
 */
export const RunViewMetricChart = ({
  metricKey,
  runInfo,
  metricsHistory,
  isLoading,
}: RunViewMetricChartProps) => {
  const loaded = metricsHistory?.[metricKey] && !isLoading;
  const isSingleMetricEntry = loaded && metricsHistory?.[metricKey].length === 1;

  const elementRef = useRef<HTMLDivElement>(null);

  const [isInViewport, setIsInViewport] = useState(false);

  // Let's use IntersectionObserver to determine if the chart is displayed within the viewport
  useEffect(() => {
    // If IntersectionObserver is not available, assume that the chart is visible
    if (!elementRef.current || !window.IntersectionObserver) {
      setIsInViewport(true);
      return () => {};
    }

    // Set the state flag only if chart is in viewport
    const intersectionObserver = new IntersectionObserver(([entry]) => {
      setIsInViewport(entry.isIntersecting);
    });

    intersectionObserver.observe(elementRef.current);
    return () => intersectionObserver.disconnect();
  }, []);

  const getChartBody = () => {
    if (!loaded) {
      return <LegacySkeleton />;
    }
    if (!isInViewport) {
      return null;
    }
    if (isSingleMetricEntry) {
      return (
        <RunViewMetricSingleValueChart
          metricKey={metricKey}
          metricsHistory={metricsHistory}
          runInfo={runInfo}
        />
      );
    }
    return (
      <RunViewMetricHistoryChart
        metricKey={metricKey}
        metricsHistory={metricsHistory}
        runInfo={runInfo}
      />
    );
  };

  return (
    <div css={{ height: 300 }} ref={elementRef}>
      {getChartBody()}
    </div>
  );
};
