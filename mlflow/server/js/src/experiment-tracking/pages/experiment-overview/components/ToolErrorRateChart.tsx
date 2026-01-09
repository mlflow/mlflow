import React, { useMemo } from 'react';
import { WrenchIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import {
  MetricViewType,
  AggregationType,
  SpanMetricKey,
  SpanFilterKey,
  SpanType,
  SpanStatus,
  SpanDimensionKey,
  TIME_BUCKET_DIMENSION_KEY,
  createSpanFilter,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from '../hooks/useTraceMetricsQuery';
import {
  OverviewChartLoadingState,
  OverviewChartErrorState,
  OverviewChartEmptyState,
  OverviewChartHeader,
  OverviewChartContainer,
  OverviewChartTimeLabel,
  ScrollableTooltip,
  useChartXAxisProps,
  useScrollableLegendProps,
} from './OverviewChartComponents';
import { formatTimestampForTraceMetrics } from '../utils/chartUtils';
import { useOverviewChartContext } from '../OverviewChartContext';

export interface ToolErrorRateChartProps {
  /** The name of the tool to display */
  toolName: string;
  /** Optional color for the line chart */
  lineColor?: string;
  /** Pre-computed overall error rate */
  overallErrorRate: number;
}

export const ToolErrorRateChart: React.FC<ToolErrorRateChartProps> = ({ toolName, lineColor, overallErrorRate }) => {
  const { experimentId, startTimeMs, endTimeMs, timeIntervalSeconds, timeBuckets } = useOverviewChartContext();
  const { theme } = useDesignSystemTheme();
  const xAxisProps = useChartXAxisProps();
  const scrollableLegendProps = useScrollableLegendProps();

  const chartLineColor = lineColor || theme.colors.red500;

  // Filter for TOOL type spans with specific name
  const toolFilters = useMemo(
    () => [createSpanFilter(SpanFilterKey.TYPE, SpanType.TOOL), createSpanFilter(SpanFilterKey.NAME, toolName)],
    [toolName],
  );

  // Query span counts grouped by status and time bucket
  const { data, isLoading, error } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.SPANS,
    metricName: SpanMetricKey.SPAN_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    filters: toolFilters,
    dimensions: [SpanDimensionKey.SPAN_STATUS],
    timeIntervalSeconds,
  });

  const dataPoints = useMemo(() => data?.data_points || [], [data?.data_points]);

  // Group data by time bucket and calculate error rate for each bucket
  const errorRateByTimestamp = useMemo(() => {
    const bucketData = new Map<number, { error: number; total: number }>();

    for (const dp of dataPoints) {
      const timeBucket = dp.dimensions?.[TIME_BUCKET_DIMENSION_KEY];
      if (!timeBucket) continue;

      const timestampMs = new Date(timeBucket).getTime();
      const count = dp.values?.[AggregationType.COUNT] || 0;
      const status = dp.dimensions?.[SpanDimensionKey.SPAN_STATUS];

      if (!bucketData.has(timestampMs)) {
        bucketData.set(timestampMs, { error: 0, total: 0 });
      }

      const bucket = bucketData.get(timestampMs)!;
      bucket.total += count;
      if (status === SpanStatus.ERROR) {
        bucket.error += count;
      }
    }

    // Convert to error rate map
    const rateMap = new Map<number, number>();
    for (const [ts, { error, total }] of bucketData) {
      rateMap.set(ts, total > 0 ? (error / total) * 100 : 0);
    }
    return rateMap;
  }, [dataPoints]);

  // Build chart data with all time buckets
  const chartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => ({
      name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
      errorRate: errorRateByTimestamp.get(timestampMs) || 0,
    }));
  }, [timeBuckets, errorRateByTimestamp, timeIntervalSeconds]);

  // Check if we have actual data
  const hasData = dataPoints.length > 0;

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  if (!hasData) {
    return (
      <OverviewChartContainer data-testid={`tool-chart-${toolName}`}>
        <OverviewChartHeader icon={<WrenchIcon css={{ color: chartLineColor }} />} title={toolName} />
        <OverviewChartEmptyState />
      </OverviewChartContainer>
    );
  }

  return (
    <OverviewChartContainer data-testid={`tool-chart-${toolName}`}>
      <OverviewChartHeader
        icon={<WrenchIcon css={{ color: chartLineColor }} />}
        title={toolName}
        value={`${overallErrorRate.toFixed(2)}%`}
        subtitle={
          <FormattedMessage defaultMessage="overall error rate" description="Subtitle for overall tool error rate" />
        }
      />

      <OverviewChartTimeLabel />

      <div css={{ height: 200 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
            <XAxis dataKey="name" {...xAxisProps} />
            <YAxis domain={[0, 100]} tickFormatter={(value) => `${value}%`} {...xAxisProps} />
            <Tooltip
              content={<ScrollableTooltip formatter={(value) => [`${value.toFixed(2)}%`, 'Error Rate']} />}
              cursor={{ stroke: theme.colors.actionTertiaryBackgroundHover }}
            />
            <Legend iconType="plainline" {...scrollableLegendProps} />
            <Line
              type="monotone"
              dataKey="errorRate"
              name="Error Rate"
              stroke={chartLineColor}
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </OverviewChartContainer>
  );
};
