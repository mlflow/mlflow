import React, { useCallback } from 'react';
import { useDesignSystemTheme, ChartLineIcon, Button } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, ReferenceArea } from 'recharts';
import { useTraceRequestsChartData } from '../hooks/useTraceRequestsChartData';
import {
  OverviewChartLoadingState,
  OverviewChartErrorState,
  OverviewChartEmptyState,
  OverviewChartHeader,
  OverviewChartTimeLabel,
  OverviewChartContainer,
  ScrollableTooltip,
  useChartXAxisProps,
  useChartZoomSelectionProps,
} from './OverviewChartComponents';

export const TraceRequestsChart: React.FC = () => {
  const { theme } = useDesignSystemTheme();
  const xAxisProps = useChartXAxisProps();
  const zoomSelectionProps = useChartZoomSelectionProps();

  // Fetch and process requests chart data (includes zoom state)
  const { totalRequests, avgRequests, isLoading, error, hasData, zoom } = useTraceRequestsChartData();
  const { zoomedData, isZoomed, refAreaLeft, refAreaRight, handleMouseDown, handleMouseMove, handleMouseUp, zoomOut } =
    zoom;

  const tooltipFormatter = useCallback((value: number) => [`${value}`, 'Requests'] as [string, string], []);

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  return (
    <OverviewChartContainer>
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <OverviewChartHeader
          icon={<ChartLineIcon />}
          title={<FormattedMessage defaultMessage="Requests" description="Title for the trace requests chart" />}
          value={totalRequests.toLocaleString()}
        />
        {isZoomed && (
          <Button componentId="mlflow.charts.trace_requests.zoom_out" size="small" onClick={zoomOut}>
            <FormattedMessage defaultMessage="Zoom Out" description="Button to reset chart zoom" />
          </Button>
        )}
      </div>
      <OverviewChartTimeLabel />

      {/* Chart */}
      <div css={{ height: 200, userSelect: 'none' }}>
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={zoomedData}
              margin={{ top: 10, right: 20, left: 0, bottom: 0 }}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
            >
              <XAxis dataKey="name" {...xAxisProps} />
              <YAxis hide />
              <Tooltip
                content={<ScrollableTooltip formatter={tooltipFormatter} />}
                cursor={{ fill: theme.colors.actionTertiaryBackgroundHover }}
              />
              <Bar dataKey="count" fill={theme.colors.blue400} radius={[4, 4, 0, 0]} />
              {avgRequests > 0 && (
                <ReferenceLine
                  y={avgRequests}
                  stroke={theme.colors.textSecondary}
                  strokeDasharray="4 4"
                  label={{
                    value: `AVG (${Math.round(avgRequests).toLocaleString()})`,
                    position: 'insideTopRight',
                    fill: theme.colors.textSecondary,
                    fontSize: 10,
                  }}
                />
              )}
              {/* Selection highlight area for zoom */}
              {refAreaLeft && refAreaRight && (
                <ReferenceArea x1={refAreaLeft} x2={refAreaRight} {...zoomSelectionProps} />
              )}
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <OverviewChartEmptyState />
        )}
      </div>
    </OverviewChartContainer>
  );
};
