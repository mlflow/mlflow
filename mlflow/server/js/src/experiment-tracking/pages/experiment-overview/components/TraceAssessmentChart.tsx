import React, { useCallback } from 'react';
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
  ScrollableTooltip,
  useChartXAxisProps,
  useChartYAxisProps,
  useScrollableLegendProps,
  DEFAULT_CHART_CONTENT_HEIGHT,
  getTracesFilteredUrl,
  getTracesFilteredByTimeRangeUrl,
  createAssessmentExistsFilter,
  createAssessmentEqualsFilter,
} from './OverviewChartComponents';
import { getLineDotStyle } from '../utils/chartUtils';
import { useOverviewChartContext } from '../OverviewChartContext';
import { useMonitoringFilters } from '../../../hooks/useMonitoringFilters';
import { useNavigate } from '../../../../common/utils/RoutingUtils';

/** Local component for chart panel with label */
const ChartPanel: React.FC<{ label: React.ReactNode; children: React.ReactElement }> = ({ label, children }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ flex: 1 }}>
      <Typography.Text color="secondary" size="sm">
        {label}
      </Typography.Text>
      <div css={{ height: DEFAULT_CHART_CONTENT_HEIGHT, marginTop: theme.spacing.sm }}>
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
  /** Optional pre-computed average value (to avoid redundant queries). If undefined, moving average chart is hidden. */
  avgValue?: number;
}

export const TraceAssessmentChart: React.FC<TraceAssessmentChartProps> = ({ assessmentName, lineColor, avgValue }) => {
  const { theme } = useDesignSystemTheme();
  const xAxisProps = useChartXAxisProps();
  const yAxisProps = useChartYAxisProps();
  const scrollableLegendProps = useScrollableLegendProps();
  const { experimentId, timeIntervalSeconds } = useOverviewChartContext();
  const [monitoringFilters] = useMonitoringFilters();
  const navigate = useNavigate();

  // Use provided color or default to green
  const chartLineColor = lineColor || theme.colors.green500;

  const distributionTooltipFormatter = useCallback((value: number) => [value, 'count'] as [number, string], []);

  // Handle click on tooltip link to navigate to traces filtered by this assessment score
  const handleViewTraces = useCallback(
    (scoreValue: string | undefined) => {
      if (!scoreValue) return;
      const url = getTracesFilteredUrl(experimentId, monitoringFilters, [
        createAssessmentEqualsFilter(assessmentName, scoreValue),
      ]);
      navigate(url);
    },
    [experimentId, assessmentName, monitoringFilters, navigate],
  );

  const timeSeriestooltipFormatter = useCallback(
    (value: number) => [value.toFixed(2), assessmentName] as [string, string],
    [assessmentName],
  );

  // Handle click on time series tooltip link to navigate to traces filtered by time AND assessment exists
  const handleViewTimeSeriesTraces = useCallback(
    (_label: string | undefined, dataPoint?: { timestampMs?: number }) => {
      if (dataPoint?.timestampMs === undefined) return;
      const url = getTracesFilteredByTimeRangeUrl(experimentId, dataPoint.timestampMs, timeIntervalSeconds, [
        createAssessmentExistsFilter(assessmentName),
      ]);
      navigate(url);
    },
    [experimentId, timeIntervalSeconds, assessmentName, navigate],
  );

  const timeSeriestooltipContent = (
    <ScrollableTooltip
      formatter={timeSeriestooltipFormatter}
      linkConfig={{
        componentId: 'mlflow.overview.quality.assessment_timeseries.view_traces_link',
        onLinkClick: handleViewTimeSeriesTraces,
      }}
    />
  );

  const distributionTooltipContent = (
    <ScrollableTooltip
      formatter={distributionTooltipFormatter}
      linkConfig={{
        componentId: 'mlflow.overview.quality.assessment.view_traces_link',
        linkText: (
          <FormattedMessage
            defaultMessage="View traces with this score"
            description="Link text to navigate to traces filtered by assessment score"
          />
        ),
        onLinkClick: handleViewTraces,
      }}
    />
  );

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

      {/* Charts side by side: distribution always shown, moving average only for numeric assessments */}
      <div css={{ display: 'flex', gap: theme.spacing.lg, marginTop: theme.spacing.sm }}>
        {/* Left: Distribution bar chart (always shown) */}
        <ChartPanel
          label={
            <FormattedMessage
              defaultMessage="Total aggregate scores"
              description="Label for assessment score distribution chart"
            />
          }
        >
          <BarChart data={distributionChartData} layout="vertical" margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
            <XAxis type="number" allowDecimals={false} {...xAxisProps} />
            <YAxis type="category" dataKey="name" {...yAxisProps} width={60} />
            <Tooltip
              content={distributionTooltipContent}
              cursor={{ fill: theme.colors.actionTertiaryBackgroundHover }}
            />
            <Legend {...scrollableLegendProps} />
            <Bar dataKey="count" fill={chartLineColor} radius={[0, 4, 4, 0]} />
          </BarChart>
        </ChartPanel>

        {/* Right: Time series line chart (only for numeric assessments with avgValue) */}
        {avgValue !== undefined && (
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
              <YAxis {...yAxisProps} />
              <Tooltip
                content={timeSeriestooltipContent}
                cursor={{ stroke: theme.colors.actionTertiaryBackgroundHover }}
              />
              <Legend {...scrollableLegendProps} />
              <Line
                type="monotone"
                dataKey="value"
                name={assessmentName}
                stroke={chartLineColor}
                strokeWidth={2}
                dot={getLineDotStyle(chartLineColor)}
                legendType="plainline"
              />
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
            </LineChart>
          </ChartPanel>
        )}
      </div>
    </OverviewChartContainer>
  );
};
