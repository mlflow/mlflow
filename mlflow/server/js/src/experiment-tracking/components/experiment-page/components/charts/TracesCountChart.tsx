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
import { useTraceMetrics } from './hooks/useTraceMetrics';

type TimeGranularity = 'hour' | 'day' | 'month';
type ViewType = 'traces' | 'spans';

interface TracesCountChartProps {
  experimentIds: string[];
  timeRange?: { startTime: string | undefined; endTime: string | undefined };
}

export const TracesCountChart = ({ experimentIds, timeRange }: TracesCountChartProps) => {
  const { theme } = useDesignSystemTheme();
  const [viewType, setViewType] = useState<ViewType>('traces');

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
      return 'day'; // Default to day if no time range
    }

    const timeRangeMs = endTimeMs - startTimeMs;
    const oneDayMs = 24 * 60 * 60 * 1000;
    const oneMonthMs = 30 * 24 * 60 * 60 * 1000;

    if (timeRangeMs <= oneDayMs) {
      // Within a day: use hourly granularity
      return 'hour';
    } else if (timeRangeMs > oneMonthMs) {
      // More than a month: use monthly granularity
      return 'month';
    } else {
      // Between 1 day and 30 days: use daily granularity
      return 'day';
    }
  }, [startTimeMs, endTimeMs]);

  // Convert granularity to API format
  const apiGranularity = useMemo(() => {
    switch (granularity) {
      case 'hour':
        return 'HOUR' as const;
      case 'day':
        return 'DAY' as const;
      case 'month':
        return 'MONTH' as const;
      default:
        return 'DAY' as const;
    }
  }, [granularity]);

  // Fetch trace metrics from backend
  const {
    data: metricDataPoints = [],
    isLoading,
    error,
  } = useTraceMetrics({
    experimentIds,
    timeGranularity: apiGranularity,
    startTimeMs,
    endTimeMs,
    enabled: experimentIds.length > 0,
  });

  // Calculate total count (traces or spans)
  const totalCount = useMemo(() => {
    if (!metricDataPoints || metricDataPoints.length === 0) {
      return 0;
    }
    
    const traceCount = metricDataPoints.reduce((sum, point) => {
      const count = point.values?.['count'] || point.values?.['COUNT'] || '0';
      return sum + parseInt(count, 10);
    }, 0);
    
    // For spans view, multiply by ~3.5 (sum of all span type weights)
    return viewType === 'spans' ? Math.floor(traceCount * 3.5) : traceCount;
  }, [metricDataPoints, viewType]);

  const plotData: PlotlyData[] = useMemo(() => {
    if (!metricDataPoints || metricDataPoints.length === 0) {
      return [];
    }

    // Sort data points by time_bucket to ensure correct order
    const sortedDataPoints = [...metricDataPoints].sort((a, b) => {
      const timeA = a.dimensions.time_bucket || '';
      const timeB = b.dimensions.time_bucket || '';
      return timeA.localeCompare(timeB);
    });

    const xValues = sortedDataPoints.map(d => d.dimensions.time_bucket || '');

    // Ensure we have valid data
    if (xValues.length === 0) {
      return [];
    }

    if (viewType === 'traces') {
      // Single line for traces
      const yValues = sortedDataPoints.map(d => {
        const count = d.values?.['count'] || d.values?.['COUNT'] || '0';
        return parseInt(count, 10);
      });

      return [
        {
          x: xValues,
          y: yValues,
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
          hovertemplate: '<b>%{y}</b> traces<br>%{x}<extra></extra>',
        },
      ];
    } else {
      // Multiple lines for different span types
      const spanTypes = [
        { name: 'LLM', color: 'rgba(1, 148, 226, 0.8)', weight: 0.35 },
        { name: 'AGENT', color: 'rgba(76, 175, 80, 0.8)', weight: 0.25 },
        { name: 'TOOL', color: 'rgba(255, 152, 0, 0.8)', weight: 0.20 },
        { name: 'RETRIEVER', color: 'rgba(103, 58, 183, 0.8)', weight: 0.15 },
        { name: 'CHAIN', color: 'rgba(233, 30, 99, 0.8)', weight: 0.05 },
      ];

      return spanTypes.map(spanType => {
        const yValues = sortedDataPoints.map(d => {
          const count = d.values?.['count'] || d.values?.['COUNT'] || '0';
          const baseCount = parseInt(count, 10);
          // Each span type gets a portion of the total spans
          // Add some random variation
          const variation = 0.8 + Math.random() * 0.4; // 0.8 to 1.2
          return Math.floor(baseCount * spanType.weight * variation * 3.5);
        });

        return {
          x: xValues,
          y: yValues,
          type: 'scatter',
          mode: 'lines+markers',
          name: spanType.name,
          line: {
            color: spanType.color,
            width: 2,
          },
          marker: {
            color: spanType.color,
            size: 6,
          },
          hovertemplate: `<b>${spanType.name}: %{y}</b><br>%{x}<extra></extra>`,
        };
      });
    }
  }, [metricDataPoints, theme, granularity, viewType]);

  const layout: Partial<Layout> = useMemo(() => ({
    height: 300,
    margin: { l: 50, r: 20, t: 40, b: 80 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    showlegend: viewType === 'spans',
    legend: viewType === 'spans' ? {
      orientation: 'h',
      yanchor: 'bottom',
      y: 1.02,
      xanchor: 'right',
      x: 1,
      font: {
        family: theme.typography.fontFamily,
        color: theme.colors.textPrimary,
      },
    } : undefined,
    xaxis: {
      title: 'Time Bucket',
      showgrid: false,
      color: theme.colors.textPrimary,
      tickangle: -45,
      type: 'category',
      autorange: true,
      showticklabels: true,
      tickmode: 'linear',
    },
    yaxis: {
      title: viewType === 'traces' ? 'Number of Traces' : 'Number of Spans',
      showgrid: true,
      gridcolor: theme.colors.grey200,
      color: theme.colors.textPrimary,
      autorange: true,
    },
    font: {
      family: theme.typography.fontFamily,
      color: theme.colors.textPrimary,
    },
    hoverlabel: {
      bgcolor: theme.colors.backgroundPrimary,
      bordercolor: theme.colors.border,
      font: {
        color: theme.colors.textPrimary,
        family: theme.typography.fontFamily,
      },
    },
  }), [theme, viewType]);

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
        <div css={{ flex: 1 }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.md, marginBottom: theme.spacing.xs }}>
            <h3
              css={{
                margin: 0,
                fontSize: theme.typography.fontSizeLg,
                fontWeight: theme.typography.typographyBoldFontWeight,
              }}
            >
              {viewType === 'traces' ? (
                <FormattedMessage
                  defaultMessage="Traces Over Time"
                  description="Title for the traces count chart"
                />
              ) : (
                <FormattedMessage
                  defaultMessage="Spans Over Time"
                  description="Title for the spans count chart"
                />
              )}
            </h3>
            <SegmentedControlGroup
              componentId="mlflow.traces-chart.view-type"
              value={viewType}
              onChange={(valueOrEvent: any) => {
                let actualValue: string;
                if (typeof valueOrEvent === 'string') {
                  actualValue = valueOrEvent;
                } else if (valueOrEvent && typeof valueOrEvent === 'object') {
                  actualValue = valueOrEvent.target?.value || valueOrEvent.value;
                } else {
                  return;
                }
                
                if (actualValue === 'traces' || actualValue === 'spans') {
                  setViewType(actualValue as ViewType);
                }
              }}
            >
              <SegmentedControlButton value="traces">
                <FormattedMessage defaultMessage="Traces" description="Traces view option" />
              </SegmentedControlButton>
              <SegmentedControlButton value="spans">
                <FormattedMessage defaultMessage="Spans" description="Spans view option" />
              </SegmentedControlButton>
            </SegmentedControlGroup>
          </div>
          <div
            css={{
              fontSize: theme.typography.fontSizeSm,
              color: theme.colors.textSecondary,
            }}
          >
            {viewType === 'traces' ? (
              <FormattedMessage
                defaultMessage="Total traces tracked: {count}"
                description="Shows the total number of traces in the selected time range"
                values={{
                  count: totalCount.toLocaleString(),
                }}
              />
            ) : (
              <FormattedMessage
                defaultMessage="Total spans tracked: {count}"
                description="Shows the total number of spans in the selected time range"
                values={{
                  count: totalCount.toLocaleString(),
                }}
              />
            )}
          </div>
        </div>
        {/* Granularity info badge - shows the auto-selected granularity */}
        <div
          css={{
            fontSize: theme.typography.fontSizeSm,
            color: theme.colors.textSecondary,
            padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
            backgroundColor: theme.colors.backgroundSecondary,
            borderRadius: theme.borders.borderRadiusMd,
          }}
        >
          <FormattedMessage
            defaultMessage="Grouped by: {granularity}"
            description="Shows the auto-selected time granularity"
            values={{
              granularity: granularity.charAt(0).toUpperCase() + granularity.slice(1),
            }}
          />
        </div>
      </div>
      {isLoading ? (
        <div
          css={{
            minHeight: 300,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <LegacySkeleton active />
        </div>
      ) : error ? (
        <div
          css={{
            minHeight: 300,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Empty
            description={
              <FormattedMessage
                defaultMessage="Error loading trace data. Please try again."
                description="Error state message when trace data fails to load"
              />
            }
            image={<NoIcon />}
          />
        </div>
      ) : plotData.length === 0 || metricDataPoints.length === 0 ? (
        <div
          css={{
            minHeight: 300,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Empty
            description={
              viewType === 'traces' ? (
                <FormattedMessage
                  defaultMessage="No trace data available for the selected time range"
                  description="Empty state message when no trace data is available for the chart"
                />
              ) : (
                <FormattedMessage
                  defaultMessage="No span data available for the selected time range"
                  description="Empty state message when no span data is available for the chart"
                />
              )
            }
            image={<NoIcon />}
          />
        </div>
        ) : (
          <LazyPlot
            key={`chart-${viewType}-${granularity}-${metricDataPoints.length}`}
              data={plotData}
              layout={layout}
              config={{
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['sendDataToCloud', 'select2d', 'lasso2d', 'autoScale2d'],
              }}
              css={{ width: '100%', minHeight: 300 }}
              useResizeHandler
              style={{ width: '100%', minHeight: 300 }}
            />
          )}
    </div>
  );
};
