import React from 'react';
import { CheckCircleIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from 'recharts';
import { useTraceAssessmentChartData } from '../hooks/useTraceAssessmentChartData';
import {
  OverviewChartLoadingState,
  OverviewChartErrorState,
  OverviewChartEmptyState,
  OverviewChartHeader,
  OverviewChartContainer,
  useChartTooltipStyle,
  useChartXAxisProps,
  useScrollableLegendProps,
} from './OverviewChartComponents';

/** Local component for chart panel with label */
const ChartPanel: React.FC<{ label: React.ReactNode; children: React.ReactElement }> = ({ label, children }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ flex: 1 }}>
      <Typography.Text color="secondary" size="sm">
        {label}
      </Typography.Text>
      <div css={{ height: 200, marginTop: theme.spacing.sm }}>
        <ResponsiveContainer width="100%" height="100%">
          {children}
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export interface TraceAssessmentChartProps {
  /** The name of the assessment to display (e.g., "Correctness", "Relevance") */
  assessmentName: string;
  /** Optional color for the line chart. Defaults to green. */
  lineColor?: string;
  /** Optional pre-computed average value (to avoid redundant queries) */
  avgValue?: number;
}

export const TraceAssessmentChart: React.FC<TraceAssessmentChartProps> = ({ assessmentName, lineColor, avgValue }) => {
  const { theme } = useDesignSystemTheme();
  const tooltipStyle = useChartTooltipStyle();
  const xAxisProps = useChartXAxisProps();
  const scrollableLegendProps = useScrollableLegendProps();

  // Use provided color or default to green
  const chartLineColor = lineColor || theme.colors.green500;

  // Fetch and process all chart data using the custom hook
  const { timeSeriesChartData, distributionChartData, isLoading, error, hasData } =
    useTraceAssessmentChartData(assessmentName);

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  if (!hasData) {
    return (
      <OverviewChartContainer>
        <OverviewChartHeader icon={<CheckCircleIcon css={{ color: chartLineColor }} />} title={assessmentName} />
        <OverviewChartEmptyState />
      </OverviewChartContainer>
    );
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

      {/* Two charts side by side */}
      <div css={{ display: 'flex', gap: theme.spacing.lg, marginTop: theme.spacing.sm }}>
        {/* Left: Distribution bar chart */}
        <ChartPanel
          label={
            <FormattedMessage
              defaultMessage="Total aggregate scores"
              description="Label for assessment score distribution chart"
            />
          }
        >
          <BarChart data={distributionChartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
            <XAxis dataKey="name" {...xAxisProps} />
            <YAxis allowDecimals={false} {...xAxisProps} />
            <Tooltip
              contentStyle={tooltipStyle}
              cursor={{ fill: theme.colors.actionTertiaryBackgroundHover }}
              formatter={(value: number) => [value, 'count']}
            />
            <Legend {...scrollableLegendProps} />
            <Bar dataKey="count" fill={chartLineColor} radius={[4, 4, 0, 0]} />
          </BarChart>
        </ChartPanel>

        {/* Right: Time series line chart */}
        <ChartPanel
          label={
            <FormattedMessage
              defaultMessage="Moving average over time"
              description="Label for assessment score over time chart"
            />
          }
        >
          <LineChart data={timeSeriesChartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
            <XAxis dataKey="name" {...xAxisProps} />
            <YAxis hide />
            <Tooltip
              contentStyle={tooltipStyle}
              cursor={{ stroke: theme.colors.actionTertiaryBackgroundHover }}
              formatter={(value: number) => [value.toFixed(2), assessmentName]}
            />
            <Legend {...scrollableLegendProps} />
            <Line
              type="monotone"
              dataKey="value"
              name={assessmentName}
              stroke={chartLineColor}
              strokeWidth={2}
              dot={false}
              legendType="plainline"
            />
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
        </ChartPanel>
      </div>
    </OverviewChartContainer>
  );
};
