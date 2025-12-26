import React, { useMemo } from 'react';
import {
  Typography,
  useDesignSystemTheme,
  Spinner,
  WrenchIcon,
  CheckCircleIcon,
  ClockIcon,
  XCircleFillIcon,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import {
  MetricViewType,
  AggregationType,
  SpanMetricKey,
  SpanFilterKey,
  SpanType,
  SpanStatus,
  SpanDimensionKey,
  createSpanFilter,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from '../hooks/useTraceMetricsQuery';
import type { OverviewChartProps } from '../types';

interface StatCardProps {
  icon: React.ReactNode;
  iconColor: string;
  iconBgColor: string;
  value: string | number;
  label: React.ReactNode;
  isLoading?: boolean;
}

const StatCard: React.FC<StatCardProps> = ({ icon, iconColor, iconBgColor, value, label, isLoading }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.md,
        padding: theme.spacing.lg,
        backgroundColor: theme.colors.backgroundPrimary,
        borderRadius: theme.borders.borderRadiusMd,
        border: `1px solid ${theme.colors.border}`,
        flex: 1,
        minWidth: 200,
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: 40,
          height: 40,
          borderRadius: theme.borders.borderRadiusMd,
          backgroundColor: iconBgColor,
          color: iconColor,
          flexShrink: 0,
        }}
      >
        {icon}
      </div>
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
        {isLoading ? (
          <Spinner size="small" />
        ) : (
          <Typography.Title level={2} css={{ margin: 0 }}>
            {value}
          </Typography.Title>
        )}
        <Typography.Text color="secondary" size="sm">
          {label}
        </Typography.Text>
      </div>
    </div>
  );
};

/**
 * Formats latency in seconds to a human-readable string
 */
function formatLatency(ms: number): string {
  if (ms < 1000) {
    return `${ms.toFixed(2)}ms`;
  }
  return `${(ms / 1000).toFixed(2)}s`;
}

/**
 * Formats a large number with K/M suffix
 */
function formatCount(count: number): string {
  if (count >= 1000000) {
    return `${(count / 1000000).toFixed(1)}M`;
  }
  if (count >= 1000) {
    return `${(count / 1000).toFixed(count >= 10000 ? 0 : 1)}K`;
  }
  return count.toLocaleString();
}

export const ToolCallStatistics: React.FC<Omit<OverviewChartProps, 'timeIntervalSeconds' | 'timeBuckets'>> = ({
  experimentId,
  startTimeMs,
  endTimeMs,
}) => {
  const { theme } = useDesignSystemTheme();

  // Filter for TOOL type spans
  const toolFilter = useMemo(() => [createSpanFilter(SpanFilterKey.TYPE, SpanType.TOOL)], []);

  // Query tool call counts grouped by status (combines total and success/error counts)
  const { data: countByStatusData, isLoading: isLoadingCounts } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.SPANS,
    metricName: SpanMetricKey.SPAN_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    filters: toolFilter,
    dimensions: [SpanDimensionKey.SPAN_STATUS],
  });

  // Query average latency for tool calls
  const { data: latencyData, isLoading: isLoadingLatency } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.SPANS,
    metricName: SpanMetricKey.LATENCY,
    aggregations: [{ aggregation_type: AggregationType.AVG }],
    filters: toolFilter,
  });

  // Calculate statistics from grouped data
  const { totalCalls, successCalls, failedCalls } = useMemo(() => {
    if (!countByStatusData?.data_points) {
      return { totalCalls: 0, successCalls: 0, failedCalls: 0 };
    }

    let total = 0;
    let success = 0;
    let failed = 0;

    for (const dp of countByStatusData.data_points) {
      const count = dp.values?.[AggregationType.COUNT] || 0;
      const status = dp.dimensions?.[SpanDimensionKey.SPAN_STATUS];
      total += count;
      if (status === SpanStatus.OK) {
        success += count;
      } else if (status === SpanStatus.ERROR) {
        failed += count;
      }
    }

    return { totalCalls: total, successCalls: success, failedCalls: failed };
  }, [countByStatusData?.data_points]);

  const successRate = totalCalls > 0 ? (successCalls / totalCalls) * 100 : 0;
  const avgLatency = latencyData?.data_points?.[0]?.values?.[AggregationType.AVG] || 0;

  const isLoading = isLoadingCounts || isLoadingLatency;

  return (
    <div
      css={{
        display: 'flex',
        gap: theme.spacing.md,
        flexWrap: 'wrap',
      }}
    >
      <StatCard
        icon={<WrenchIcon />}
        iconColor={theme.colors.blue600}
        iconBgColor={theme.colors.blue100}
        value={formatCount(totalCalls)}
        label={
          <FormattedMessage defaultMessage="Total Tool Calls" description="Label for total tool calls statistic" />
        }
        isLoading={isLoading}
      />
      <StatCard
        icon={<CheckCircleIcon />}
        iconColor={theme.colors.green600}
        iconBgColor={theme.colors.green100}
        value={`${successRate.toFixed(2)}%`}
        label={<FormattedMessage defaultMessage="Success Rate" description="Label for success rate statistic" />}
        isLoading={isLoading}
      />
      <StatCard
        icon={<ClockIcon />}
        iconColor={theme.colors.yellow600}
        iconBgColor={theme.colors.yellow100}
        value={formatLatency(avgLatency)}
        label={<FormattedMessage defaultMessage="Avg Latency" description="Label for average latency statistic" />}
        isLoading={isLoading}
      />
      <StatCard
        icon={<XCircleFillIcon />}
        iconColor={theme.colors.red600}
        iconBgColor={theme.colors.red100}
        value={formatCount(failedCalls)}
        label={<FormattedMessage defaultMessage="Failed Calls" description="Label for failed calls statistic" />}
        isLoading={isLoading}
      />
    </div>
  );
};
