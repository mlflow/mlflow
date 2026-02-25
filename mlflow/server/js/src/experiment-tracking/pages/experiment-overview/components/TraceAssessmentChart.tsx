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
  useClickableTooltip,
  LockedTooltipOverlay,
} from './OverviewChartComponents';
import { getLineDotStyle } from '../utils/chartUtils';
import { useOverviewChartContext } from '../OverviewChartContext';
import { useMonitoringFilters } from '../../../hooks/useMonitoringFilters';
import { useNavigate } from '../../../../common/utils/RoutingUtils';

export interface TraceAssessmentChartProps {
  assessmentName: string;
  lineColor?: string;
  avgValue?: number;
}

export const TraceAssessmentChart: React.FC<TraceAssessmentChartProps> = ({ assessmentName, lineColor, avgValue }) => {
  const { theme } = useDesignSystemTheme();
  const xAxisProps = useChartXAxisProps();
  const yAxisProps = useChartYAxisProps();
  const scrollableLegendProps = useScrollableLegendProps();
  const { experimentIds, timeIntervalSeconds } = useOverviewChartContext();
  const [monitoringFilters] = useMonitoringFilters();
  const navigate = useNavigate();

  const distributionTooltip = useClickableTooltip();
  const timeSeriesTooltip = useClickableTooltip();

  const chartLineColor = lineColor || theme.colors.green500;

  const distributionTooltipFormatter = useCallback((value: number) => [value, 'count'] as [number, string], []);
  const timeSeriestooltipFormatter = useCallback(
    (value: number) => [value.toFixed(2), assessmentName] as [string, string],
    [assessmentName],
  );

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

  const onDistributionChartClick = useCallback(
    (data: any, event: React.MouseEvent) => distributionTooltip.handleChartClick(data, event),
    [distributionTooltip],
  );

  const onTimeSeriesChartClick = useCallback(
    (data: any, event: React.MouseEvent) => timeSeriesTooltip.handleChartClick(data, event),
    [timeSeriesTooltip],
  );

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

      <div css={{ display: 'flex', gap: theme.spacing.lg, marginTop: theme.spacing.sm }}>
        {/* Left: Distribution bar chart */}
        <div css={{ flex: 1 }}>
          <Typography.Text color="secondary" size="sm">
            <FormattedMessage
              defaultMessage="Total aggregate scores"
              description="Label for assessment score distribution chart"
            />
          </Typography.Text>
          <div
            ref={distributionTooltip.containerRef}
            css={{ height: DEFAULT_CHART_CONTENT_HEIGHT, marginTop: theme.spacing.sm, position: 'relative' }}
          >
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={distributionChartData}
                layout="vertical"
                margin={{ top: 10, right: 10, left: 10, bottom: 0 }}
                onClick={onDistributionChartClick}
              >
                <XAxis type="number" allowDecimals={false} {...xAxisProps} />
                <YAxis type="category" dataKey="name" {...yAxisProps} width={60} />
                <Tooltip
                  content={
                    distributionTooltip.isLocked ? () => null : (
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
                    )
                  }
                  cursor={{ fill: theme.colors.actionTertiaryBackgroundHover }}
                />
                <Legend {...scrollableLegendProps} />
                <Bar dataKey="count" fill={chartLineColor} radius={[0, 4, 4, 0]} cursor="pointer" />
              </BarChart>
            </ResponsiveContainer>
            {distributionTooltip.lockedTooltip && (
              <LockedTooltipOverlay
                data={distributionTooltip.lockedTooltip}
                tooltipRef={distributionTooltip.tooltipRef}
                formatter={distributionTooltipFormatter}
                linkConfig={{
                  componentId: 'mlflow.overview.quality.assessment.view_traces_link',
                  linkText: (
                    <FormattedMessage
                      defaultMessage="View traces with this score"
                      description="Link text to navigate to traces filtered by assessment score"
                    />
                  ),
                  onLinkClick: () => handleViewTraces(distributionTooltip.lockedTooltip?.label),
                }}
              />
            )}
          </div>
        </div>

        {/* Right: Time series line chart */}
        {avgValue !== undefined && (
          <div css={{ flex: 1 }}>
            <Typography.Text color="secondary" size="sm">
              <FormattedMessage
                defaultMessage="Moving average over time"
                description="Label for assessment score over time chart"
              />
            </Typography.Text>
            <div
              ref={timeSeriesTooltip.containerRef}
              css={{ height: DEFAULT_CHART_CONTENT_HEIGHT, marginTop: theme.spacing.sm, position: 'relative' }}
            >
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={timeSeriesChartData}
                  margin={{ top: 10, right: 10, left: 10, bottom: 0 }}
                  onClick={onTimeSeriesChartClick}
                >
                  <XAxis dataKey="name" {...xAxisProps} />
                  <YAxis {...yAxisProps} />
                  <Tooltip
                    content={
                      timeSeriesTooltip.isLocked ? () => null : (
                        <ScrollableTooltip
                          formatter={timeSeriestooltipFormatter}
                          linkConfig={{
                            componentId: 'mlflow.overview.quality.assessment_timeseries.view_traces_link',
                            onLinkClick: handleViewTimeSeriesTraces,
                          }}
                        />
                      )
                    }
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
                    cursor="pointer"
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
              </ResponsiveContainer>
              {timeSeriesTooltip.lockedTooltip && (
                <LockedTooltipOverlay
                  data={timeSeriesTooltip.lockedTooltip}
                  tooltipRef={timeSeriesTooltip.tooltipRef}
                  formatter={timeSeriestooltipFormatter}
                  linkConfig={{
                    componentId: 'mlflow.overview.quality.assessment_timeseries.view_traces_link',
                    onLinkClick: () =>
                      handleViewTimeSeriesTraces(
                        timeSeriesTooltip.lockedTooltip?.label,
                        timeSeriesTooltip.lockedTooltip?.payload?.[0]?.payload,
                      ),
                  }}
                />
              )}
            </div>
          </div>
        )}
      </div>
    </OverviewChartContainer>
  );
};
