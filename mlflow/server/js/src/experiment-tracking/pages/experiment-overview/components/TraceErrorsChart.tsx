import React, { useMemo, useState } from 'react';
import { useDesignSystemTheme, DangerIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ComposedChart, Bar, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, ReferenceLine } from 'recharts';
import {
  MetricViewType,
  AggregationType,
  TraceMetricKey,
  TIME_BUCKET_DIMENSION_KEY,
  TraceFilterKey,
  TraceStatus,
  createTraceFilter,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from '../hooks/useTraceMetricsQuery';
import { formatTimestampForTraceMetrics, getTimestampFromDataPoint } from '../utils/chartUtils';
import {
  OverviewChartLoadingState,
  OverviewChartErrorState,
  OverviewChartEmptyState,
  OverviewChartHeader,
  OverviewChartTimeLabel,
} from './OverviewChartComponents';
import type { OverviewChartProps } from '../types';

// Filter to get only error traces
const ERROR_FILTER = createTraceFilter(TraceFilterKey.STATUS, TraceStatus.ERROR);

export const TraceErrorsChart: React.FC<OverviewChartProps> = ({
  experimentId,
  startTimeMs,
  endTimeMs,
  timeIntervalSeconds,
}) => {
  const { theme } = useDesignSystemTheme();

  // Fetch error count metrics grouped by time bucket
  const {
    data: errorCountData,
    isLoading: isLoadingErrors,
    error: errorCountError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    timeIntervalSeconds,
    filters: [ERROR_FILTER],
  });

  // Fetch total trace count metrics grouped by time bucket (for calculating error rate)
  // This query is also used by TraceRequestsChart, so React Query will dedupe it
  const {
    data: totalCountData,
    isLoading: isLoadingTotal,
    error: totalCountError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    timeIntervalSeconds,
  });

  const errorDataPoints = useMemo(() => errorCountData?.data_points || [], [errorCountData?.data_points]);
  const totalDataPoints = useMemo(() => totalCountData?.data_points || [], [totalCountData?.data_points]);
  const isLoading = isLoadingErrors || isLoadingTotal;
  const error = errorCountError || totalCountError;

  // Calculate totals by summing time-bucketed data
  const totalErrors = useMemo(
    () => errorDataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.COUNT] || 0), 0),
    [errorDataPoints],
  );
  const totalTraces = useMemo(
    () => totalDataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.COUNT] || 0), 0),
    [totalDataPoints],
  );
  const overallErrorRate = totalTraces > 0 ? (totalErrors / totalTraces) * 100 : 0;

  // Create a map of time bucket -> total count for easy lookup
  const totalCountMap = useMemo(() => {
    const map = new Map<string, number>();
    for (const dp of totalDataPoints) {
      const timeBucket = dp.dimensions?.[TIME_BUCKET_DIMENSION_KEY];
      if (timeBucket) {
        map.set(timeBucket, dp.values?.[AggregationType.COUNT] || 0);
      }
    }
    return map;
  }, [totalDataPoints]);

  // Prepare chart data for recharts - combine error count and error rate
  const chartData = useMemo(() => {
    return errorDataPoints.map((dp) => {
      const timestampMs = getTimestampFromDataPoint(dp);
      const timeBucket = dp.dimensions?.[TIME_BUCKET_DIMENSION_KEY] || '';
      const errorCount = dp.values?.[AggregationType.COUNT] || 0;
      const totalCount = totalCountMap.get(timeBucket) || 0;
      const errorRate = totalCount > 0 ? (errorCount / totalCount) * 100 : 0;

      return {
        name: timestampMs ? formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds) : '',
        errorCount,
        errorRate: Math.round(errorRate * 100) / 100, // Round to 2 decimal places
      };
    });
  }, [errorDataPoints, totalCountMap, timeIntervalSeconds]);

  // Calculate average error rate across time buckets for the reference line
  const avgErrorRate = useMemo(() => {
    if (chartData.length === 0) return 0;
    const sum = chartData.reduce((acc, dp) => acc + dp.errorRate, 0);
    return sum / chartData.length;
  }, [chartData]);

  // Track hovered legend item
  const [hoveredSeries, setHoveredSeries] = useState<string | null>(null);

  // Get opacity based on hover state
  const getOpacity = (seriesKey: string) => {
    if (hoveredSeries === null) return 1;
    return hoveredSeries === seriesKey ? 1 : 0.2;
  };

  // Handle legend hover
  const handleLegendMouseEnter = (data: { value: string }) => {
    setHoveredSeries(data.value);
  };

  const handleLegendMouseLeave = () => {
    setHoveredSeries(null);
  };

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  return (
    <div
      css={{
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        padding: theme.spacing.lg,
        backgroundColor: theme.colors.backgroundPrimary,
      }}
    >
      <OverviewChartHeader
        icon={<DangerIcon />}
        title={<FormattedMessage defaultMessage="Errors" description="Title for the errors chart" />}
        value={totalErrors.toLocaleString()}
        subtitle={`(Overall error rate: ${overallErrorRate.toFixed(1)}%)`}
      />

      <OverviewChartTimeLabel />

      {/* Chart */}
      <div css={{ height: 200, marginTop: theme.spacing.sm }}>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
              <XAxis
                dataKey="name"
                tick={{ fontSize: 10, fill: theme.colors.textSecondary, dy: theme.spacing.sm }}
                axisLine={false}
                tickLine={false}
                interval="preserveStartEnd"
              />
              <YAxis yAxisId="left" hide />
              <YAxis yAxisId="right" hide />
              <Tooltip
                contentStyle={{
                  backgroundColor: theme.colors.backgroundPrimary,
                  border: `1px solid ${theme.colors.border}`,
                  borderRadius: theme.borders.borderRadiusMd,
                  fontSize: 12,
                }}
                cursor={{ fill: theme.colors.actionTertiaryBackgroundHover }}
                formatter={(value: number, name: string) => {
                  if (name === 'Error Count') {
                    return [value.toLocaleString(), name];
                  }
                  return [`${value.toFixed(1)}%`, name];
                }}
              />
              <Bar
                yAxisId="left"
                dataKey="errorCount"
                fill={theme.colors.red400}
                radius={[4, 4, 0, 0]}
                name="Error Count"
                fillOpacity={getOpacity('Error Count')}
                legendType="square"
              />
              {avgErrorRate > 0 && (
                <ReferenceLine
                  yAxisId="right"
                  y={avgErrorRate}
                  stroke={theme.colors.textSecondary}
                  strokeDasharray="4 4"
                  label={{
                    value: `AVG (${avgErrorRate.toFixed(1)}%)`,
                    position: 'insideTopRight',
                    fill: theme.colors.textSecondary,
                    fontSize: 10,
                  }}
                />
              )}
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="errorRate"
                stroke={theme.colors.yellow500}
                strokeWidth={2}
                dot={false}
                name="Error Rate"
                strokeOpacity={getOpacity('Error Rate')}
                legendType="plainline"
              />
              <Legend
                verticalAlign="bottom"
                height={36}
                onMouseEnter={handleLegendMouseEnter}
                onMouseLeave={handleLegendMouseLeave}
                formatter={(value) => (
                  <span
                    style={{
                      color: theme.colors.textPrimary,
                      fontSize: theme.typography.fontSizeSm,
                      cursor: 'pointer',
                    }}
                  >
                    {value}
                  </span>
                )}
              />
            </ComposedChart>
          </ResponsiveContainer>
        ) : (
          <OverviewChartEmptyState />
        )}
      </div>
    </div>
  );
};
