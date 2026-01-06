import React from 'react';
import {
  useDesignSystemTheme,
  WrenchIcon,
  CheckCircleIcon,
  ClockIcon,
  XCircleFillIcon,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useToolCallStatisticsData } from '../hooks/useToolCallStatisticsData';
import { formatCount, formatLatency } from '../utils/chartUtils';
import { StatCard } from './OverviewLayoutComponents';
import type { OverviewChartProps } from '../types';

export const ToolCallStatistics: React.FC<Omit<OverviewChartProps, 'timeIntervalSeconds' | 'timeBuckets'>> = ({
  experimentId,
  startTimeMs,
  endTimeMs,
}) => {
  const { theme } = useDesignSystemTheme();

  // Fetch and process tool call statistics using the custom hook
  const { totalCalls, failedCalls, successRate, avgLatency, isLoading } = useToolCallStatisticsData({
    experimentId,
    startTimeMs,
    endTimeMs,
  });

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
