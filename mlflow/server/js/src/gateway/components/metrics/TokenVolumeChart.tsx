import React, { useMemo } from 'react';
import { useDesignSystemTheme, LightningIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
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
  formatCount,
  formatTimestamp,
} from './MetricsChartComponents';
import { useLegendHighlight } from './useLegendHighlight';

const SECONDS_PER_DAY = 86400;

interface TokenVolumeChartProps {
  endpointId?: string;
  startTime?: number;
  endTime?: number;
  bucketSize?: number; // Size in seconds (e.g., 3600 for hourly, 86400 for daily)
}

export const TokenVolumeChart: React.FC<TokenVolumeChartProps> = ({
  endpointId,
  startTime,
  endTime,
  bucketSize = SECONDS_PER_DAY,
}) => {
  const { theme } = useDesignSystemTheme();
  const tooltipStyle = useChartTooltipStyle();
  const xAxisProps = useChartXAxisProps();
  const legendFormatter = useChartLegendFormatter();
  const { getOpacity, handleLegendMouseEnter, handleLegendMouseLeave } = useLegendHighlight(0.8, 0.2);

  const { data, isLoading, error } = useUsageMetricsQuery({
    endpoint_id: endpointId,
    start_time: startTime,
    end_time: endTime,
    bucket_size: bucketSize,
  });

  const { chartData, totalTokens, totalInputTokens, totalOutputTokens, hasData } = useMemo(() => {
    const metrics = data?.metrics || [];
    if (metrics.length === 0) {
      return {
        chartData: [],
        totalTokens: 0,
        totalInputTokens: 0,
        totalOutputTokens: 0,
        hasData: false,
      };
    }

    let totalInput = 0;
    let totalOutput = 0;

    const chartData = metrics.map((m) => {
      totalInput += m.total_prompt_tokens;
      totalOutput += m.total_completion_tokens;
      return {
        name: formatTimestamp(m.time_bucket, bucketSize),
        inputTokens: m.total_prompt_tokens,
        outputTokens: m.total_completion_tokens,
      };
    });

    return {
      chartData,
      totalTokens: totalInput + totalOutput,
      totalInputTokens: totalInput,
      totalOutputTokens: totalOutput,
      hasData: true,
    };
  }, [data, bucketSize]);

  const areaColors = {
    inputTokens: theme.colors.blue400,
    outputTokens: theme.colors.green400,
  };

  if (isLoading) {
    return <MetricsChartLoadingState />;
  }

  if (error) {
    return <MetricsChartErrorState />;
  }

  return (
    <MetricsChartContainer>
      <MetricsChartHeader
        icon={<LightningIcon />}
        title={<FormattedMessage defaultMessage="Token Volume" description="Title for the token volume chart" />}
        value={formatCount(totalTokens)}
        subtitle={`(${formatCount(totalInputTokens)} input, ${formatCount(totalOutputTokens)} output)`}
      />

      <MetricsChartTimeLabel />

      <div css={{ height: 200, marginTop: theme.spacing.sm }}>
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 30, bottom: 0 }}>
              <XAxis dataKey="name" {...xAxisProps} />
              <YAxis hide />
              <Tooltip
                contentStyle={tooltipStyle}
                cursor={{ fill: theme.colors.actionTertiaryBackgroundHover }}
                formatter={(value: number, name: string) => [formatCount(value), name]}
              />
              <Area
                type="monotone"
                dataKey="inputTokens"
                stackId="1"
                stroke={areaColors.inputTokens}
                fill={areaColors.inputTokens}
                strokeOpacity={getOpacity('Input Tokens')}
                fillOpacity={getOpacity('Input Tokens')}
                strokeWidth={2}
                name="Input Tokens"
              />
              <Area
                type="monotone"
                dataKey="outputTokens"
                stackId="1"
                stroke={areaColors.outputTokens}
                fill={areaColors.outputTokens}
                strokeOpacity={getOpacity('Output Tokens')}
                fillOpacity={getOpacity('Output Tokens')}
                strokeWidth={2}
                name="Output Tokens"
              />
              <Legend
                verticalAlign="bottom"
                iconType="plainline"
                height={36}
                onMouseEnter={handleLegendMouseEnter}
                onMouseLeave={handleLegendMouseLeave}
                formatter={legendFormatter}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <MetricsChartEmptyState />
        )}
      </div>
    </MetricsChartContainer>
  );
};
