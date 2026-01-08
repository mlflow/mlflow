import React from 'react';
import { useDesignSystemTheme, ChartLineIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { useTraceRequestsChartData } from '../hooks/useTraceRequestsChartData';
import {
  OverviewChartLoadingState,
  OverviewChartErrorState,
  OverviewChartEmptyState,
  OverviewChartHeader,
  OverviewChartTimeLabel,
  OverviewChartContainer,
  useChartTooltipStyle,
  useChartXAxisProps,
} from './OverviewChartComponents';
import type { OverviewChartProps } from '../types';

export const TraceRequestsChart: React.FC<OverviewChartProps> = (props) => {
  const { theme } = useDesignSystemTheme();
  const tooltipStyle = useChartTooltipStyle();
  const xAxisProps = useChartXAxisProps();

  // Fetch and process requests chart data
  const { chartData, totalRequests, isLoading, error, hasData } = useTraceRequestsChartData(props);

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  return (
    <OverviewChartContainer>
      <OverviewChartHeader
        icon={<ChartLineIcon />}
        title={<FormattedMessage defaultMessage="Requests" description="Title for the trace requests chart" />}
        value={totalRequests.toLocaleString()}
      />
      <OverviewChartTimeLabel />

      {/* Chart */}
      <div css={{ height: 200 }}>
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
              <XAxis dataKey="name" {...xAxisProps} />
              <YAxis hide />
              <Tooltip
                contentStyle={tooltipStyle}
                cursor={{ fill: theme.colors.actionTertiaryBackgroundHover }}
                formatter={(value: number) => [`${value}`, 'Requests']}
              />
              <Bar dataKey="count" fill={theme.colors.blue400} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <OverviewChartEmptyState />
        )}
      </div>
    </OverviewChartContainer>
  );
};
