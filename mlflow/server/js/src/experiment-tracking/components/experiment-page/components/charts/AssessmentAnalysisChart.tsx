import { useMemo, useState } from 'react';
import {
  useDesignSystemTheme,
  Empty,
  NoIcon,
  LegacySkeleton,
  SegmentedControlGroup,
  SegmentedControlButton,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import type { Data as PlotlyData, Layout } from 'plotly.js';
import { LazyPlot } from '../../../LazyPlot';

type TimeGranularity = 'hour' | 'day' | 'month';
type AggregationMetric = 'average' | 'p50' | 'p75' | 'p90' | 'p95' | 'p99';

interface AssessmentAnalysisChartProps {
  experimentIds: string[];
  timeRange?: { startTime: string | undefined; endTime: string | undefined };
}

export const AssessmentAnalysisChart = ({ experimentIds, timeRange }: AssessmentAnalysisChartProps) => {
  const { theme } = useDesignSystemTheme();
  const [selectedAssessment, setSelectedAssessment] = useState<string>('correctness');
  const [aggregationMetric, setAggregationMetric] = useState<AggregationMetric>('average');

  // Convert timeRange strings to milliseconds
  const startTimeMs = useMemo(() => {
    if (!timeRange?.startTime) return undefined;
    const parsed = parseInt(timeRange.startTime, 10);
    return isNaN(parsed) ? undefined : parsed;
  }, [timeRange?.startTime]);

  const endTimeMs = useMemo(() => {
    if (!timeRange?.endTime) return undefined;
    const parsed = parseInt(timeRange.endTime, 10);
    return isNaN(parsed) ? undefined : parsed;
  }, [timeRange?.endTime]);

  // Automatically determine granularity based on time range
  const granularity = useMemo((): TimeGranularity => {
    if (!startTimeMs || !endTimeMs) {
      return 'day';
    }

    const timeRangeMs = endTimeMs - startTimeMs;
    const oneDayMs = 24 * 60 * 60 * 1000;
    const oneMonthMs = 30 * 24 * 60 * 60 * 1000;

    if (timeRangeMs <= oneDayMs) {
      return 'hour';
    } else if (timeRangeMs > oneMonthMs) {
      return 'month';
    } else {
      return 'day';
    }
  }, [startTimeMs, endTimeMs]);

  // Available assessment names
  const assessmentNames = useMemo(() => {
    return ['correctness', 'relevance', 'toxicity', 'groundedness', 'coherence'];
  }, []);

  // Mock data for total aggregate scores (distribution)
  const aggregateScoresData = useMemo(() => {
    // Create bins from 0 to 1 with 0.1 intervals
    const bins = [
      { label: '[0, 0.1)', count: Math.floor(Math.random() * 50) + 10 },
      { label: '[0.1, 0.2)', count: Math.floor(Math.random() * 30) + 5 },
      { label: '[0.2, 0.3)', count: Math.floor(Math.random() * 40) + 10 },
      { label: '[0.3, 0.4)', count: Math.floor(Math.random() * 50) + 15 },
      { label: '[0.4, 0.5)', count: Math.floor(Math.random() * 60) + 20 },
      { label: '[0.5, 0.6)', count: Math.floor(Math.random() * 80) + 30 },
      { label: '[0.6, 0.7)', count: Math.floor(Math.random() * 100) + 40 },
      { label: '[0.7, 0.8)', count: Math.floor(Math.random() * 150) + 60 },
      { label: '[0.8, 0.9)', count: Math.floor(Math.random() * 200) + 100 },
      { label: '[0.9, 1]', count: Math.floor(Math.random() * 300) + 150 },
    ];
    return bins;
  }, [selectedAssessment]);

  // Mock data for scores over time
  const scoresOverTimeData = useMemo(() => {
    if (!startTimeMs || !endTimeMs) return [];

    const timeRangeMs = endTimeMs - startTimeMs;
    let intervalMs: number;
    let numPoints: number;

    switch (granularity) {
      case 'hour':
        intervalMs = 60 * 60 * 1000;
        numPoints = Math.min(Math.floor(timeRangeMs / intervalMs), 24);
        break;
      case 'day':
        intervalMs = 24 * 60 * 60 * 1000;
        numPoints = Math.min(Math.floor(timeRangeMs / intervalMs), 30);
        break;
      case 'month':
        intervalMs = 30 * 24 * 60 * 60 * 1000;
        numPoints = Math.min(Math.floor(timeRangeMs / intervalMs), 12);
        break;
    }

    const dataPoints = [];
    const startDate = new Date(startTimeMs);
    
    if (granularity === 'hour') {
      startDate.setMinutes(0, 0, 0);
    } else if (granularity === 'day') {
      startDate.setHours(0, 0, 0, 0);
    } else {
      startDate.setDate(1);
      startDate.setHours(0, 0, 0, 0);
    }

    let currentTimestamp = startDate.getTime();

    for (let i = 0; i < numPoints; i++) {
      const date = new Date(currentTimestamp);
      let timeBucket: string;

      if (granularity === 'hour') {
        timeBucket = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')} ${String(date.getHours()).padStart(2, '0')}:00`;
      } else if (granularity === 'day') {
        timeBucket = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
      } else {
        timeBucket = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
      }

      // Generate scores (0-1 range, typically higher for positive assessments)
      const baseScore = {
        average: 0.75,
        p50: 0.70,
        p75: 0.85,
        p90: 0.92,
        p95: 0.96,
        p99: 0.99,
      }[aggregationMetric];

      // Add random variation
      const variation = -0.15 + Math.random() * 0.2; // -0.15 to 0.05
      const score = Math.max(0, Math.min(1, baseScore + variation));

      dataPoints.push({
        time_bucket: timeBucket,
        score: score,
      });

      if (granularity === 'month') {
        date.setMonth(date.getMonth() + 1);
        currentTimestamp = date.getTime();
      } else {
        currentTimestamp += intervalMs;
      }
    }

    return dataPoints;
  }, [startTimeMs, endTimeMs, granularity, aggregationMetric, selectedAssessment]);

  const isLoading = false;
  const error = null;

  // Plot data for aggregate scores (bar chart)
  const aggregatePlotData: PlotlyData[] = useMemo(() => {
    if (aggregateScoresData.length === 0) return [];

    return [
      {
        x: aggregateScoresData.map(d => d.label),
        y: aggregateScoresData.map(d => d.count),
        type: 'bar',
        marker: {
          color: 'rgba(1, 148, 226, 0.7)',
        },
        hovertemplate: '<b>%{y}</b> assessments<br>%{x}<extra></extra>',
      },
    ];
  }, [aggregateScoresData]);

  // Plot data for scores over time (line chart)
  const timeSeriesPlotData: PlotlyData[] = useMemo(() => {
    if (scoresOverTimeData.length === 0) return [];

    return [
      {
        x: scoresOverTimeData.map(d => d.time_bucket),
        y: scoresOverTimeData.map(d => d.score),
        type: 'scatter',
        mode: 'lines+markers',
        line: {
          color: 'rgba(1, 148, 226, 0.8)',
          width: 2,
        },
        marker: {
          color: 'rgba(1, 148, 226, 0.8)',
          size: 6,
        },
        hovertemplate: '<b>%{y:.2f}</b><br>%{x}<extra></extra>',
      },
    ];
  }, [scoresOverTimeData]);

  const aggregateLayout: Partial<Layout> = useMemo(() => ({
    height: 300,
    margin: { l: 50, r: 20, t: 10, b: 80 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    autosize: true,
    xaxis: {
      title: 'Score Range',
      showgrid: false,
      color: theme.colors.textPrimary,
      tickangle: -45,
    },
    yaxis: {
      title: 'Count',
      showgrid: true,
      gridcolor: theme.colors.grey200,
      color: theme.colors.textPrimary,
    },
    font: {
      family: theme.typography.fontFamily,
      color: theme.colors.textPrimary,
    },
  }), [theme]);

  const timeSeriesLayout: Partial<Layout> = useMemo(() => ({
    height: 300,
    margin: { l: 50, r: 20, t: 10, b: 80 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    autosize: true,
    xaxis: {
      title: 'Time',
      showgrid: false,
      color: theme.colors.textPrimary,
      tickangle: -45,
      type: 'category',
    },
    yaxis: {
      title: 'Score',
      showgrid: true,
      gridcolor: theme.colors.grey200,
      color: theme.colors.textPrimary,
      range: [0, 1],
    },
    font: {
      family: theme.typography.fontFamily,
      color: theme.colors.textPrimary,
    },
  }), [theme]);

  const metricLabel = useMemo(() => {
    switch (aggregationMetric) {
      case 'average': return 'Average';
      case 'p50': return 'P50';
      case 'p75': return 'P75';
      case 'p90': return 'P90';
      case 'p95': return 'P95';
      case 'p99': return 'P99';
    }
  }, [aggregationMetric]);

  const totalAssessments = useMemo(() => {
    return aggregateScoresData.reduce((sum, d) => sum + d.count, 0);
  }, [aggregateScoresData]);

  return (
    <div
      css={{
        backgroundColor: theme.colors.backgroundPrimary,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        padding: theme.spacing.md,
      }}
    >
      <div
        css={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          marginBottom: theme.spacing.md,
        }}
      >
        <div>
          <h3
            css={{
              margin: 0,
              fontSize: theme.typography.fontSizeLg,
              fontWeight: theme.typography.typographyBoldFontWeight,
            }}
          >
            <FormattedMessage
              defaultMessage="Assessment Analysis"
              description="Title for the assessment analysis chart"
            />
          </h3>
          <div
            css={{
              fontSize: theme.typography.fontSizeSm,
              color: theme.colors.textSecondary,
              marginTop: theme.spacing.xs,
            }}
          >
            <FormattedMessage
              defaultMessage="Total assessments: {count}"
              description="Shows the total number of assessments"
              values={{
                count: totalAssessments.toLocaleString(),
              }}
            />
          </div>
        </div>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <span
            css={{
              fontSize: theme.typography.fontSizeXs,
              color: theme.colors.textSecondary,
              whiteSpace: 'nowrap',
            }}
          >
            Assessment:
          </span>
          <select
            value={selectedAssessment}
            onChange={(e) => setSelectedAssessment(e.target.value)}
            css={{
              fontSize: theme.typography.fontSizeSm,
              padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.borders.borderRadiusMd,
              backgroundColor: theme.colors.backgroundPrimary,
              color: theme.colors.textPrimary,
              cursor: 'pointer',
              '&:hover': {
                borderColor: theme.colors.primary,
              },
              '&:focus': {
                outline: 'none',
                borderColor: theme.colors.primary,
              },
            }}
          >
            {assessmentNames.map(name => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
        </div>
      </div>

      {isLoading ? (
        <div css={{ minHeight: 300 }}>
          <LegacySkeleton active />
        </div>
      ) : error ? (
        <Empty
          description={
            <FormattedMessage
              defaultMessage="Error loading assessment data. Please try again."
              description="Error state message"
            />
          }
          image={<NoIcon />}
        />
      ) : (
        <div
          css={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: theme.spacing.md,
            '@media (max-width: 1200px)': {
              gridTemplateColumns: '1fr',
            },
          }}
        >
          {/* Left: Total aggregate scores */}
          <div>
            <div
              css={{
                fontSize: theme.typography.fontSizeMd,
                fontWeight: theme.typography.typographyBoldFontWeight,
                color: theme.colors.textPrimary,
                marginBottom: theme.spacing.sm,
              }}
            >
              <FormattedMessage
                defaultMessage="Total Aggregate Scores"
                description="Title for aggregate scores distribution"
              />
            </div>
            {aggregatePlotData.length > 0 ? (
              <LazyPlot
                key={`aggregate-${selectedAssessment}`}
                data={aggregatePlotData}
                layout={aggregateLayout}
                config={{
                  displayModeBar: true,
                  displaylogo: false,
                  modeBarButtonsToRemove: ['sendDataToCloud', 'select2d', 'lasso2d', 'autoScale2d'],
                }}
                css={{ width: '100%' }}
                useResizeHandler
                style={{ width: '100%' }}
              />
            ) : (
              <Empty
                description={
                  <FormattedMessage
                    defaultMessage="No aggregate data available"
                    description="Empty state for aggregate scores"
                  />
                }
                image={<NoIcon />}
              />
            )}
          </div>

          {/* Right: Scores over time */}
          <div>
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, marginBottom: theme.spacing.sm }}>
              <div
                css={{
                  fontSize: theme.typography.fontSizeMd,
                  fontWeight: theme.typography.typographyBoldFontWeight,
                  color: theme.colors.textPrimary,
                }}
              >
                <FormattedMessage
                  defaultMessage="{metric} Over Time"
                  description="Title for scores over time"
                  values={{
                    metric: metricLabel,
                  }}
                />
              </div>
              <SegmentedControlGroup
                componentId="mlflow.assessment-analysis.metric"
                value={aggregationMetric}
                onChange={(valueOrEvent: any) => {
                  let actualValue: string;
                  if (typeof valueOrEvent === 'string') {
                    actualValue = valueOrEvent;
                  } else if (valueOrEvent && typeof valueOrEvent === 'object') {
                    actualValue = valueOrEvent.target?.value || valueOrEvent.value;
                  } else {
                    return;
                  }
                  
                  if (['average', 'p50', 'p75', 'p90', 'p95', 'p99'].includes(actualValue)) {
                    setAggregationMetric(actualValue as AggregationMetric);
                  }
                }}
              >
                <SegmentedControlButton value="average">Avg</SegmentedControlButton>
                <SegmentedControlButton value="p50">P50</SegmentedControlButton>
                <SegmentedControlButton value="p75">P75</SegmentedControlButton>
                <SegmentedControlButton value="p90">P90</SegmentedControlButton>
                <SegmentedControlButton value="p95">P95</SegmentedControlButton>
                <SegmentedControlButton value="p99">P99</SegmentedControlButton>
              </SegmentedControlGroup>
            </div>
            {timeSeriesPlotData.length > 0 ? (
              <LazyPlot
                key={`timeseries-${selectedAssessment}-${aggregationMetric}-${granularity}`}
                data={timeSeriesPlotData}
                layout={timeSeriesLayout}
                config={{
                  displayModeBar: true,
                  displaylogo: false,
                  modeBarButtonsToRemove: ['sendDataToCloud', 'select2d', 'lasso2d', 'autoScale2d'],
                }}
                css={{ width: '100%' }}
                useResizeHandler
                style={{ width: '100%' }}
              />
            ) : (
              <Empty
                description={
                  <FormattedMessage
                    defaultMessage="No time series data available"
                    description="Empty state for time series"
                  />
                }
                image={<NoIcon />}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
};

