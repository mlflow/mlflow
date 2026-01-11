import React, { useMemo } from 'react';
import { useDesignSystemTheme, DangerIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ComposedChart, Bar, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, ReferenceLine } from 'recharts';
import { useUsageMetricsQuery } from '../../hooks/useUsageMetricsQuery';
import {
  MetricsChartLoadingState,
  MetricsChartErrorState,
  MetricsChartEmptyState,
  MetricsChartHeader,
  MetricsChartContainer,
  MetricsChartTimeLabel,
  useChartTooltipStyle,
  useChartXAxisProps,
  useChartLegendFormatter,
  formatTimestamp,
} from './MetricsChartComponents';
import { useLegendHighlight } from './useLegendHighlight';

const SECONDS_PER_DAY = 86400;

interface ErrorRateChartProps {
  endpointId?: string;
  startTime?: number;
  endTime?: number;
  bucketSize?: number; // Size in seconds (e.g., 3600 for hourly, 86400 for daily)
}

export const ErrorRateChart: React.FC<ErrorRateChartProps> = ({
  endpointId,
  startTime,
  endTime,
  bucketSize = SECONDS_PER_DAY,
}) => {
  const { theme } = useDesignSystemTheme();
  const tooltipStyle = useChartTooltipStyle();
  const xAxisProps = useChartXAxisProps();
  const legendFormatter = useChartLegendFormatter();
  const { getOpacity, handleLegendMouseEnter, handleLegendMouseLeave } = useLegendHighlight();

  const { data, isLoading, error } = useUsageMetricsQuery({
    endpoint_id: endpointId,
    start_time: startTime,
    end_time: endTime,
    bucket_size: bucketSize,
  });

  const { chartData, totalErrors, overallErrorRate, avgErrorRate, hasData } = useMemo(() => {
    const metrics = data?.metrics || [];
    if (metrics.length === 0) {
      return {
        chartData: [],
        totalErrors: 0,
        overallErrorRate: 0,
        avgErrorRate: 0,
        hasData: false,
      };
    }

    let totalFailed = 0;
    let totalInvocations = 0;
    let errorRateSum = 0;

    const chartData = metrics.map((m) => {
      totalFailed += m.failed_invocations;
      totalInvocations += m.total_invocations;
      errorRateSum += m.error_rate * 100; // Convert from 0-1 to percentage
      return {
        name: formatTimestamp(m.time_bucket, bucketSize),
        errorCount: m.failed_invocations,
        errorRate: m.error_rate * 100, // Convert from 0-1 to percentage
      };
    });

    const overallErrorRate = totalInvocations > 0 ? (totalFailed / totalInvocations) * 100 : 0;
    const avgErrorRate = metrics.length > 0 ? errorRateSum / metrics.length : 0;

    return {
      chartData,
      totalErrors: totalFailed,
      overallErrorRate,
      avgErrorRate,
      hasData: true,
    };
  }, [data, bucketSize]);

  if (isLoading) {
    return <MetricsChartLoadingState />;
  }

  if (error) {
    return <MetricsChartErrorState />;
  }

  return (
    <MetricsChartContainer>
      <MetricsChartHeader
        icon={<DangerIcon />}
        title={<FormattedMessage defaultMessage="Error Rate" description="Title for the error rate chart" />}
        value={totalErrors.toLocaleString()}
        subtitle={`(Overall error rate: ${overallErrorRate.toFixed(1)}%)`}
      />

      <MetricsChartTimeLabel />

      <div css={{ height: 200, marginTop: theme.spacing.sm }}>
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
              <XAxis dataKey="name" {...xAxisProps} />
              <YAxis yAxisId="left" hide />
              <YAxis yAxisId="right" domain={[0, 100]} hide />
              <Tooltip
                contentStyle={tooltipStyle}
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
                formatter={legendFormatter}
              />
            </ComposedChart>
          </ResponsiveContainer>
        ) : (
          <MetricsChartEmptyState />
        )}
      </div>
    </MetricsChartContainer>
  );
};
