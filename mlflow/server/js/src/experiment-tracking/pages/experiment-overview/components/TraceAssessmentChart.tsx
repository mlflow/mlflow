import React from 'react';
import { CheckCircleIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { useTraceAssessmentChartData } from '../hooks/useTraceAssessmentChartData';
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

export interface TraceAssessmentChartProps extends OverviewChartProps {
  /** The name of the assessment to display (e.g., "Correctness", "Relevance") */
  assessmentName: string;
  /** Optional color for the line chart. Defaults to green. */
  lineColor?: string;
  /** Optional pre-computed average value (to avoid redundant queries) */
  avgValue?: number;
}

export const TraceAssessmentChart: React.FC<TraceAssessmentChartProps> = ({
  assessmentName,
  lineColor,
  avgValue,
  ...chartProps
}) => {
  const { theme } = useDesignSystemTheme();
  const tooltipStyle = useChartTooltipStyle();
  const xAxisProps = useChartXAxisProps();

  // Use provided color or default to green
  const chartLineColor = lineColor || theme.colors.green500;

  // Fetch and process assessment chart data
  const { chartData, isLoading, error, hasData } = useTraceAssessmentChartData({
    ...chartProps,
    assessmentName,
  });

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  return (
    <OverviewChartContainer data-testid={`assessment-chart-${assessmentName}`}>
      <OverviewChartHeader
        icon={<CheckCircleIcon css={{ color: chartLineColor }} />}
        title={assessmentName}
        value={avgValue !== undefined ? avgValue.toFixed(2) : undefined}
        subtitle={
          avgValue !== undefined ? (
            <FormattedMessage defaultMessage="avg score" description="Subtitle for average assessment score" />
          ) : undefined
        }
      />

      <OverviewChartTimeLabel />

      {/* Chart */}
      <div css={{ height: 200, marginTop: theme.spacing.sm }}>
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 30, left: 30, bottom: 0 }}>
              <XAxis dataKey="name" {...xAxisProps} />
              <YAxis hide />
              <Tooltip
                contentStyle={tooltipStyle}
                cursor={{ stroke: theme.colors.actionTertiaryBackgroundHover }}
                formatter={(value: number) => [value.toFixed(2), assessmentName]}
              />
              <Line type="monotone" dataKey="value" stroke={chartLineColor} strokeWidth={2} dot={false} />
              {avgValue !== undefined && (
                <ReferenceLine
                  y={avgValue}
                  stroke={theme.colors.textSecondary}
                  strokeDasharray="4 4"
                  label={{
                    value: `AVG (${avgValue.toFixed(2)})`,
                    position: 'insideTopRight',
                    fill: theme.colors.textSecondary,
                    fontSize: 10,
                  }}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <OverviewChartEmptyState />
        )}
      </div>
    </OverviewChartContainer>
  );
};
