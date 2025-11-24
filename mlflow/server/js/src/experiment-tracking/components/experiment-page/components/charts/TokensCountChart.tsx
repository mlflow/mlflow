import { useMemo, useState } from 'react';
import {
  useDesignSystemTheme,
  Empty,
  NoIcon,
  LegacySkeleton,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import type { Data as PlotlyData, Layout } from 'plotly.js';
import { LazyPlot } from '../../../LazyPlot';

type TimeGranularity = 'hour' | 'day' | 'month';

interface TokensCountChartProps {
  experimentIds: string[];
  timeRange?: { startTime: string | undefined; endTime: string | undefined };
}

export const TokensCountChart = ({ experimentIds, timeRange }: TokensCountChartProps) => {
  const { theme } = useDesignSystemTheme();
  const [selectedTraceName, setSelectedTraceName] = useState<string>('all');

  // Available trace names for filtering
  const traceNames = useMemo(() => {
    return ['all', 'chat_completion', 'text_generation', 'embedding_generation', 'question_answering', 'summarization'];
  }, []);

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
      return 'hour';
    } else if (timeRangeMs > oneMonthMs) {
      return 'month';
    } else {
      return 'day';
    }
  }, [startTimeMs, endTimeMs]);

  // Mock data generation for tokens
  const mockTokenData = useMemo(() => {
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

      // Generate token counts (typically higher than trace counts, ~2000-3000 per trace)
      // Adjust base tokens based on selected trace (different trace types use different amounts)
      let baseTokens = 50000;
      if (selectedTraceName !== 'all') {
        const traceMultipliers: Record<string, number> = {
          'chat_completion': 1.2,
          'text_generation': 1.5,
          'embedding_generation': 0.5,
          'question_answering': 0.8,
          'summarization': 1.0,
        };
        baseTokens = baseTokens * (traceMultipliers[selectedTraceName] || 1.0);
      }
      
      const variation = Math.random() * 30000;
      const totalTokens = Math.floor(baseTokens + variation);
      
      // Split into input (40%) and output (60%) tokens
      const inputTokens = Math.floor(totalTokens * 0.4);
      const outputTokens = totalTokens - inputTokens;

      dataPoints.push({
        time_bucket: timeBucket,
        total_tokens: totalTokens,
        input_tokens: inputTokens,
        output_tokens: outputTokens,
      });

      if (granularity === 'month') {
        date.setMonth(date.getMonth() + 1);
        currentTimestamp = date.getTime();
      } else {
        currentTimestamp += intervalMs;
      }
    }

    return dataPoints;
  }, [startTimeMs, endTimeMs, granularity, selectedTraceName]);

  const isLoading = false;
  const error = null;

  const plotData: PlotlyData[] = useMemo(() => {
    if (mockTokenData.length === 0) {
      return [];
    }

    const xValues = mockTokenData.map(d => d.time_bucket);

    return [
      {
        x: xValues,
        y: mockTokenData.map(d => d.input_tokens),
        type: 'bar',
        name: 'Input',
        marker: {
          color: 'rgba(40, 167, 69, 0.7)',
        },
        customdata: mockTokenData.map(d => [d.output_tokens, d.total_tokens]),
        hovertemplate: '<b>%{x}</b><br>' +
                      'Input: %{y:,}<br>' +
                      'Output: %{customdata[0]:,}<br>' +
                      'Total: %{customdata[1]:,}<extra></extra>',
      },
      {
        x: xValues,
        y: mockTokenData.map(d => d.output_tokens),
        type: 'bar',
        name: 'Output',
        marker: {
          color: 'rgba(255, 193, 7, 0.7)',
        },
        customdata: mockTokenData.map(d => [d.input_tokens, d.total_tokens]),
        hovertemplate: '<b>%{x}</b><br>' +
                      'Input: %{customdata[0]:,}<br>' +
                      'Output: %{y:,}<br>' +
                      'Total: %{customdata[1]:,}<extra></extra>',
      },
    ];
  }, [mockTokenData]);

  const totalTokens = useMemo(() => {
    return mockTokenData.reduce((sum, d) => sum + d.total_tokens, 0);
  }, [mockTokenData]);

  const layout: Partial<Layout> = useMemo(() => ({
    height: 300,
    margin: { l: 50, r: 20, t: 40, b: 80 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    barmode: 'stack',
    showlegend: true,
    legend: {
      orientation: 'h',
      yanchor: 'bottom',
      y: 1.02,
      xanchor: 'right',
      x: 1,
      font: {
        family: theme.typography.fontFamily,
        color: theme.colors.textPrimary,
      },
    },
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
      title: 'Number of Tokens',
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
        <div>
          <h3
            css={{
              margin: 0,
              fontSize: theme.typography.fontSizeLg,
              fontWeight: theme.typography.typographyBoldFontWeight,
            }}
          >
            <FormattedMessage
              defaultMessage="Tokens Over Time"
              description="Title for the tokens count chart"
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
              defaultMessage="Total tokens tracked: {count}"
              description="Shows the total number of tokens in the selected time range"
              values={{
                count: totalTokens.toLocaleString(),
              }}
            />
          </div>
        </div>
        <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
            <span
              css={{
                fontSize: theme.typography.fontSizeXs,
                color: theme.colors.textSecondary,
                whiteSpace: 'nowrap',
              }}
            >
              Trace Name:
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
                minWidth: 120,
                '&:hover': {
                  borderColor: theme.colors.primary,
                },
                '&:focus': {
                  outline: 'none',
                  borderColor: theme.colors.primary,
                },
              }}
            >
              {traceNames.map((name) => (
                <option key={name} value={name}>
                  {name === 'all' ? 'All Trace Names' : name}
                </option>
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
                defaultMessage="Error loading token data. Please try again."
                description="Error state message when token data fails to load"
              />
            }
            image={<NoIcon />}
          />
        </div>
      ) : plotData.length === 0 || mockTokenData.length === 0 ? (
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
                defaultMessage="No token data available for the selected time range"
                description="Empty state message when no token data is available for the chart"
              />
            }
            image={<NoIcon />}
          />
        </div>
      ) : (
        <LazyPlot
          key={`chart-${granularity}-${selectedTraceName}-${mockTokenData.length}`}
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

