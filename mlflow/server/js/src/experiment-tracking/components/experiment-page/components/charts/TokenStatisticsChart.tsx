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
type TokenMetric = 'average' | 'p50' | 'p75' | 'p90' | 'p95' | 'p99';

interface TokenStatisticsChartProps {
  experimentIds: string[];
  timeRange?: { startTime: string | undefined; endTime: string | undefined };
}

export const TokenStatisticsChart = ({ experimentIds, timeRange }: TokenStatisticsChartProps) => {
  const { theme } = useDesignSystemTheme();
  const [selectedMetric, setSelectedMetric] = useState<TokenMetric>('average');

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

  const mockTokenStatisticsData = useMemo(() => {
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

    const baseTokensPerTrace = 2500; // Average base tokens per trace
    const p50Factor = 0.7;
    const p75Factor = 1.1;
    const p90Factor = 1.6;
    const p95Factor = 2.2;
    const p99Factor = 3.5;

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

      const variation = 0.85 + Math.random() * 0.3; // 0.85 to 1.15

      const avg = Math.floor(baseTokensPerTrace * variation);
      const p50 = Math.floor(avg * p50Factor * (0.9 + Math.random() * 0.2));
      const p75 = Math.floor(avg * p75Factor * (0.9 + Math.random() * 0.2));
      const p90 = Math.floor(avg * p90Factor * (0.9 + Math.random() * 0.2));
      const p95 = Math.floor(avg * p95Factor * (0.9 + Math.random() * 0.2));
      const p99 = Math.floor(avg * p99Factor * (0.9 + Math.random() * 0.2));

      dataPoints.push({
        time_bucket: timeBucket,
        average: avg,
        p50,
        p75,
        p90,
        p95,
        p99,
      });

      if (granularity === 'month') {
        date.setMonth(date.getMonth() + 1);
        currentTimestamp = date.getTime();
      } else {
        currentTimestamp += intervalMs;
      }
    }

    return dataPoints;
  }, [startTimeMs, endTimeMs, granularity]);

  const isLoading = false;
  const error = null;

  const plotData: PlotlyData[] = useMemo(() => {
    if (mockTokenStatisticsData.length === 0) {
      return [];
    }

    const xValues = mockTokenStatisticsData.map(d => d.time_bucket);
    const yValues = mockTokenStatisticsData.map(d => d[selectedMetric]);

    return [
      {
        x: xValues,
        y: yValues,
        type: 'scatter',
        mode: 'lines+markers',
        name: selectedMetric.charAt(0).toUpperCase() + selectedMetric.slice(1),
        line: {
          color: 'rgba(1, 148, 226, 0.8)',
          width: 2,
        },
        marker: {
          color: 'rgba(1, 148, 226, 0.8)',
          size: 6,
        },
        hovertemplate: `<b>${selectedMetric.toUpperCase()}: %{y:,.0f} tokens</b><br>%{x}<extra></extra>`,
      },
    ];
  }, [mockTokenStatisticsData, selectedMetric]);

  const currentMetricValue = useMemo(() => {
    if (mockTokenStatisticsData.length === 0) return 0;
    const sum = mockTokenStatisticsData.reduce((acc, d) => acc + d[selectedMetric], 0);
    return Math.floor(sum / mockTokenStatisticsData.length);
  }, [mockTokenStatisticsData, selectedMetric]);

  const layout: Partial<Layout> = useMemo(() => ({
    height: 300,
    margin: { l: 50, r: 20, t: 40, b: 80 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    showlegend: false,
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
      title: 'Tokens per Trace',
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
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <h3
              css={{
                margin: 0,
                fontSize: theme.typography.fontSizeLg,
                fontWeight: theme.typography.typographyBoldFontWeight,
              }}
            >
              <FormattedMessage
                defaultMessage="Token Statistics Over Time"
                description="Title for the token statistics chart"
              />
            </h3>
            <SegmentedControlGroup
              componentId="mlflow.token-statistics-chart.metric-type"
              value={selectedMetric}
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
                  setSelectedMetric(actualValue as TokenMetric);
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
              defaultMessage="{metric} tokens per trace: {value}"
              description="Shows the current selected token metric value"
              values={{
                metric: selectedMetric.charAt(0).toUpperCase() + selectedMetric.slice(1),
                value: currentMetricValue.toLocaleString(),
              }}
            />
          </div>
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
                defaultMessage="Error loading token statistics data. Please try again."
                description="Error state message when token statistics data fails to load"
              />
            }
            image={<NoIcon />}
          />
        </div>
      ) : plotData.length === 0 || mockTokenStatisticsData.length === 0 ? (
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
                defaultMessage="No token statistics data available for the selected time range"
                description="Empty state message when no token statistics data is available for the chart"
              />
            }
            image={<NoIcon />}
          />
        </div>
      ) : (
        <LazyPlot
          key={`chart-${granularity}-${selectedMetric}-${mockTokenStatisticsData.length}`}
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

