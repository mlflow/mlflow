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
import { OverviewChartErrorState } from './OverviewChartComponents';

export const ToolCallStatistics: React.FC = () => {
  const { theme } = useDesignSystemTheme();

  // Fetch and process tool call statistics using the custom hook
  const { totalCalls, failedCalls, successRate, avgLatency, isLoading, error } = useToolCallStatisticsData();

  if (error) {
    return <OverviewChartErrorState />;
  }

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
        iconColor={theme.colors.blue500}
        iconBgColor={`${theme.colors.blue500}1A`}
        value={formatCount(totalCalls)}
        label={
          <FormattedMessage defaultMessage="Total Tool Calls" description="Label for total tool calls statistic" />
        }
        isLoading={isLoading}
      />
      <StatCard
        icon={<ClockIcon />}
        iconColor={theme.colors.yellow500}
        iconBgColor={`${theme.colors.yellow500}1A`}
        value={formatLatency(avgLatency)}
        label={<FormattedMessage defaultMessage="Avg Latency" description="Label for average latency statistic" />}
        isLoading={isLoading}
      />
      <StatCard
        icon={<CheckCircleIcon />}
        iconColor={theme.colors.green500}
        iconBgColor={`${theme.colors.green500}1A`}
        value={`${successRate.toFixed(2)}%`}
        label={<FormattedMessage defaultMessage="Success Rate" description="Label for success rate statistic" />}
        isLoading={isLoading}
      />
      <StatCard
        icon={<XCircleFillIcon />}
        iconColor={theme.colors.red500}
        iconBgColor={`${theme.colors.red500}1A`}
        value={formatCount(failedCalls)}
        label={<FormattedMessage defaultMessage="Failed Calls" description="Label for failed calls statistic" />}
        isLoading={isLoading}
      />
    </div>
  );
};
