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
type LatencyMetric = 'average' | 'p50' | 'p75' | 'p90' | 'p95' | 'p99';

interface TraceLatencyChartProps {
  experimentIds: string[];
  timeRange?: { startTime: string | undefined; endTime: string | undefined };
}

export const TraceLatencyChart = ({ experimentIds, timeRange }: TraceLatencyChartProps) => {
  const { theme } = useDesignSystemTheme();
  const [latencyMetric, setLatencyMetric] = useState<LatencyMetric>('average');
  const [selectedTraceName, setSelectedTraceName] = useState<string>('all');

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

  // Available trace names for filtering
  const traceNames = useMemo(() => {
    return [
      'all',
      'chat_completion',
      'text_generation',
      'embedding_generation',
      'question_answering',
      'summarization',
      'code_generation',
    ];
  }, []);

  // Mock data generation for latency
  const mockLatencyData = useMemo(() => {
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

      // Generate latency values in milliseconds (realistic LLM latencies)
      // Base latency varies by percentile
      const baseLatency = {
        average: 800,
        p50: 600,
        p75: 1000,
        p90: 1500,
        p95: 2000,
        p99: 3500,
      }[latencyMetric];

      // Add random variation
      const variation = 0.7 + Math.random() * 0.6; // 0.7 to 1.3
      const latency = Math.floor(baseLatency * variation);

      dataPoints.push({
        time_bucket: timeBucket,
        latency: latency,
      });

      if (granularity === 'month') {
        date.setMonth(date.getMonth() + 1);
        currentTimestamp = date.getTime();
      } else {
        currentTimestamp += intervalMs;
      }
    }

    return dataPoints;
  }, [startTimeMs, endTimeMs, granularity, latencyMetric, selectedTraceName]);

  const isLoading = false;
  const error = null;

  const plotData: PlotlyData[] = useMemo(() => {
    if (mockLatencyData.length === 0) {
      return [];
    }

    const xValues = mockLatencyData.map(d => d.time_bucket);
    const yValues = mockLatencyData.map(d => d.latency);

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
        hovertemplate: '<b>%{y} ms</b><br>%{x}<extra></extra>',
      },
    ];
  }, [mockLatencyData]);

  const averageLatency = useMemo(() => {
    if (mockLatencyData.length === 0) return 0;
    const sum = mockLatencyData.reduce((acc, d) => acc + d.latency, 0);
    return Math.floor(sum / mockLatencyData.length);
  }, [mockLatencyData]);

  const layout: Partial<Layout> = useMemo(() => ({
    height: 300,
    margin: { l: 50, r: 20, t: 40, b: 80 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
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
      title: 'Latency (ms)',
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
  }), [theme]);

  const metricLabel = useMemo(() => {
    switch (latencyMetric) {
      case 'average': return 'Average';
      case 'p50': return 'P50';
      case 'p75': return 'P75';
      case 'p90': return 'P90';
      case 'p95': return 'P95';
      case 'p99': return 'P99';
    }
  }, [latencyMetric]);

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
              <FormattedMessage
                defaultMessage="Trace Latency Over Time"
                description="Title for the trace latency chart"
              />
            </h3>
            <SegmentedControlGroup
              componentId="mlflow.latency-chart.metric"
              value={latencyMetric}
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
                  setLatencyMetric(actualValue as LatencyMetric);
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
          <div
            css={{
              fontSize: theme.typography.fontSizeSm,
              color: theme.colors.textSecondary,
            }}
          >
            <FormattedMessage
              defaultMessage="{metric} latency: {value} ms"
              description="Shows the selected metric and average value"
              values={{
                metric: metricLabel,
                value: averageLatency.toLocaleString(),
              }}
            />
          </div>
        </div>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.md }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
            <span
              css={{
                fontSize: theme.typography.fontSizeXs,
                color: theme.colors.textSecondary,
                whiteSpace: 'nowrap',
              }}
            >
              Trace name:
            </span>
            <select
              value={selectedTraceName}
              onChange={(e) => setSelectedTraceName(e.target.value)}
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
              <option value="all">All</option>
              {traceNames.slice(1).map(name => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
          </div>
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
                defaultMessage="Error loading latency data. Please try again."
                description="Error state message when latency data fails to load"
              />
            }
            image={<NoIcon />}
          />
        </div>
      ) : plotData.length === 0 || mockLatencyData.length === 0 ? (
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
                defaultMessage="No latency data available for the selected time range"
                description="Empty state message when no latency data is available for the chart"
              />
            }
            image={<NoIcon />}
          />
        </div>
      ) : (
        <LazyPlot
          key={`chart-${latencyMetric}-${selectedTraceName}-${granularity}-${mockLatencyData.length}`}
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

