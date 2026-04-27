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
  // eslint-disable-next-line import/no-deprecated
  Cell,
} from 'recharts';
import type { AssessmentChartDataPoint, DistributionChartDataPoint } from '../hooks/useAssessmentChartsSectionData';
import {
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
  assessmentName: string;
  lineColor?: string;
  /** When undefined, the moving average chart is hidden (non-numeric assessments) */
  avgValue?: number;
  timeSeriesChartData: AssessmentChartDataPoint[];
  distributionChartData: DistributionChartDataPoint[];
  enableTraceNavigation?: boolean;
}

export const TraceAssessmentChart: React.FC<TraceAssessmentChartProps> = ({
  assessmentName,
  lineColor,
  avgValue,
  timeSeriesChartData,
  distributionChartData,
  enableTraceNavigation = true,
}) => {
  const { theme } = useDesignSystemTheme();
  const xAxisProps = useChartXAxisProps();
  const yAxisProps = useChartYAxisProps();
  const scrollableLegendProps = useScrollableLegendProps();
  const { experimentIds, timeIntervalSeconds } = useOverviewChartContext();
  const [monitoringFilters] = useMonitoringFilters();
  const navigate = useNavigate();

  const chartLineColor = lineColor || theme.colors.green500;

  // Map assessment value names to Tag background colors with increased opacity for chart visibility.
  // Base colors from design system Tag backgrounds (tagBackgroundLime/tagBackgroundCoral) with 5x opacity.
  const barColorLime = 'rgba(2, 179, 2, 0.40)';
  const barColorCoral = 'rgba(240, 0, 64, 0.50)';
  const getBarColor = useCallback(
    (name: string) => {
      const lower = name.toLowerCase();
      if (lower === 'yes' || lower === 'true') return barColorLime;
      if (lower === 'no' || lower === 'false') return barColorCoral;
      return chartLineColor;
    },
    [chartLineColor],
  );

  const distributionTooltipFormatter = useCallback((value: number) => [value, 'count'] as [number, string], []);

  const handleViewTraces = useCallback(
    (scoreValue: string | undefined) => {
      if (!scoreValue) return;
      const url = getTracesFilteredUrl(experimentIds[0], monitoringFilters, [
        createAssessmentEqualsFilter(assessmentName, scoreValue),
      ]);
      navigate(url);
    },
    [experimentIds, assessmentName, monitoringFilters, navigate],
  );

  const timeSeriesTooltipFormatter = useCallback(
    (value: number) => [value.toFixed(2), assessmentName] as [string, string],
    [assessmentName],
  );

  const handleViewTimeSeriesTraces = useCallback(
    (_label: string | undefined, dataPoint?: { timestampMs?: number }) => {
      if (dataPoint?.timestampMs === undefined) return;
      const url = getTracesFilteredByTimeRangeUrl(experimentIds[0], dataPoint.timestampMs, timeIntervalSeconds, [
        createAssessmentExistsFilter(assessmentName),
      ]);
      navigate(url);
    },
    [experimentIds, timeIntervalSeconds, assessmentName, navigate],
  );

  const timeSeriesTooltipContent = (
    <ScrollableTooltip
      formatter={timeSeriesTooltipFormatter}
      componentId="mlflow.overview.quality.assessment_timeseries.view_traces_link"
      linkConfig={
        enableTraceNavigation
          ? {
              onLinkClick: handleViewTimeSeriesTraces,
            }
          : undefined
      }
    />
  );

  const distributionTooltipContent = (
    <ScrollableTooltip
      formatter={distributionTooltipFormatter}
      componentId="mlflow.overview.quality.assessment.view_traces_link"
      linkConfig={
        enableTraceNavigation
          ? {
              linkText: (
                <FormattedMessage
                  defaultMessage="View traces with this score"
                  description="Link text to navigate to traces filtered by assessment score"
                />
              ),
              onLinkClick: handleViewTraces,
            }
          : undefined
      }
    />
  );

  const hasData = timeSeriesChartData.length > 0 || distributionChartData.length > 0;

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

      <div css={{ display: 'flex', gap: theme.spacing.lg, marginTop: theme.spacing.sm }}>
        <ChartPanel
          label={
            <FormattedMessage
              defaultMessage="Total aggregate scores"
              description="Label for assessment score distribution chart"
            />
          }
        >
          <BarChart
            data={distributionChartData}
            layout="vertical"
            barCategoryGap="28%"
            margin={{ top: 10, right: 10, left: 10, bottom: 0 }}
          >
            <XAxis type="number" allowDecimals={false} {...xAxisProps} />
            <YAxis
              type="category"
              dataKey="name"
              {...yAxisProps}
              tick={{ ...yAxisProps.tick, fontSize: 14 }}
              width={80}
            />
            <Tooltip
              content={distributionTooltipContent}
              cursor={{ fill: theme.colors.actionTertiaryBackgroundHover }}
              wrapperStyle={{ pointerEvents: 'auto' }}
            />
            <Legend {...scrollableLegendProps} />
            <Bar dataKey="count" fill={chartLineColor} radius={[0, 4, 4, 0]}>
              {distributionChartData.map((entry) => (
                // eslint-disable-next-line import/no-deprecated
                <Cell key={entry.name} fill={getBarColor(entry.name)} />
              ))}
            </Bar>
          </BarChart>
        </ChartPanel>

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
                content={timeSeriesTooltipContent}
                cursor={{ stroke: theme.colors.actionTertiaryBackgroundHover }}
                wrapperStyle={{ pointerEvents: 'auto' }}
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
