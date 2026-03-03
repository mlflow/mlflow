import React, { useCallback, useMemo, useState } from 'react';
import { useDesignSystemTheme, PieChartIcon, type DesignSystemThemeInterface } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Sector } from 'recharts';
import { formatCostUSD } from '@databricks/web-shared/model-trace-explorer';
import { useTraceCostBreakdownChartData } from '../hooks/useTraceCostBreakdownChartData';
import { useTraceCostDimension } from '../hooks/useTraceCostDimension';
import { CostDimensionToggle } from './CostDimensionToggle';
import {
  OverviewChartLoadingState,
  OverviewChartErrorState,
  OverviewChartEmptyState,
  OverviewChartHeader,
  OverviewChartContainer,
  DEFAULT_CHART_CONTENT_HEIGHT,
} from './OverviewChartComponents';
import {
  useChartColors,
  useLegendHighlight,
  calculatePieActiveShapeGeometry,
  type ActiveShapeProps,
} from '../utils/chartUtils';

// Pie chart sizing constants
const PIE_INNER_RADIUS = 50;
const PIE_OUTER_RADIUS = 70;
const PIE_PADDING_ANGLE = 2;

/**
 * Renders the active shape for a pie slice with outer arc, connecting line, and labels.
 */
const renderActiveShape = (props: ActiveShapeProps, theme: DesignSystemThemeInterface['theme']) => {
  const { cx, cy, innerRadius, outerRadius, startAngle, endAngle, fill, name, value, percentage } = props;
  const { sx, sy, mx, my, ex, ey, textAnchor, cos } = calculatePieActiveShapeGeometry(props, theme);

  // Clamp label Y so it stays within the chart area (top and bottom)
  const maxLabelY = cy * 2 - theme.spacing.md * 2;
  const labelY = Math.max(theme.spacing.md, Math.min(maxLabelY, ey));
  const labelX = ex + (cos >= 0 ? 1 : -1) * theme.spacing.xs;

  return (
    <g>
      {/* Main pie sector */}
      <Sector
        cx={cx}
        cy={cy}
        innerRadius={innerRadius}
        outerRadius={outerRadius}
        startAngle={startAngle}
        endAngle={endAngle}
        fill={fill}
      />
      {/* Outer highlight arc (selection indicator) */}
      <Sector
        cx={cx}
        cy={cy}
        startAngle={startAngle}
        endAngle={endAngle}
        innerRadius={outerRadius + theme.spacing.xs}
        outerRadius={outerRadius + theme.spacing.sm}
        fill={fill}
      />
      {/* Connecting line: radial segment -> horizontal segment */}
      <path d={`M${sx},${sy}L${mx},${my}L${ex},${labelY}`} stroke={fill} fill="none" />
      {/* Dot at line end */}
      <circle cx={ex} cy={labelY} r={2} fill={fill} stroke="none" />
      {/* Model name label */}
      <text
        x={labelX}
        y={labelY}
        textAnchor={textAnchor}
        fill={theme.colors.textPrimary}
        fontSize={theme.typography.fontSizeSm}
        fontWeight={500}
      >
        {name}
      </text>
      {/* Cost and percentage label */}
      <text
        x={labelX}
        y={labelY + theme.spacing.md}
        textAnchor={textAnchor}
        fill={theme.colors.textSecondary}
        fontSize={theme.typography.fontSizeSm}
      >
        {formatCostUSD(value)}, {percentage.toFixed(2)}%
      </text>
    </g>
  );
};

export const TraceCostBreakdownChart: React.FC = () => {
  const { theme } = useDesignSystemTheme();
  const { dimension, setDimension } = useTraceCostDimension();
  const { chartData, totalCost, isLoading, error, hasData } = useTraceCostBreakdownChartData(dimension);
  const { getChartColor } = useChartColors();
  const [activeIndex, setActiveIndex] = useState<number | undefined>(undefined);
  const { getOpacity, handleLegendMouseEnter, handleLegendMouseLeave } = useLegendHighlight();

  const activeShapeRenderer = useMemo(
    () => (props: unknown) => renderActiveShape(props as ActiveShapeProps, theme),
    [theme],
  );

  const onPieEnter = useCallback((_: unknown, index: number) => {
    setActiveIndex(index);
  }, []);

  const onPieLeave = useCallback(() => {
    setActiveIndex(undefined);
  }, []);

  const onLegendMouseEnter = useCallback(
    (data: { value: string | undefined }) => {
      handleLegendMouseEnter(data);
      const index = chartData.findIndex((entry) => entry.name === data.value);
      if (index !== -1) {
        setActiveIndex(index);
      }
    },
    [handleLegendMouseEnter, chartData],
  );

  const onLegendMouseLeave = useCallback(() => {
    handleLegendMouseLeave();
    setActiveIndex(undefined);
  }, [handleLegendMouseLeave]);

  const legendFormatter = useCallback(
    (value: string, entry: unknown) => {
      const typedEntry = entry as { payload?: { percentage?: number } };
      return (
        <span
          css={{
            color: theme.colors.textPrimary,
            fontSize: theme.typography.fontSizeSm,
          }}
        >
          {value}
          <span css={{ color: theme.colors.textSecondary, marginLeft: theme.spacing.xs }}>
            {typedEntry.payload?.percentage?.toFixed(0)}%
          </span>
        </span>
      );
    },
    [theme],
  );

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  return (
    <OverviewChartContainer componentId="mlflow.charts.trace_cost_breakdown">
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <OverviewChartHeader
          icon={<PieChartIcon />}
          title={<FormattedMessage defaultMessage="Cost Breakdown" description="Title for the cost breakdown chart" />}
          value={formatCostUSD(totalCost)}
          subtitle={
            <FormattedMessage defaultMessage="Total Cost" description="Subtitle for the cost breakdown chart total" />
          }
        />
        <div css={{ flexShrink: 0 }}>
          <CostDimensionToggle
            componentId="mlflow.charts.trace_cost_breakdown.dimension"
            value={dimension}
            onChange={setDimension}
          />
        </div>
      </div>

      <div css={{ height: DEFAULT_CHART_CONTENT_HEIGHT, marginTop: theme.spacing.sm, overflow: 'visible' }}>
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <PieChart style={{ overflow: 'visible' }}>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                innerRadius={PIE_INNER_RADIUS}
                outerRadius={PIE_OUTER_RADIUS}
                paddingAngle={PIE_PADDING_ANGLE}
                dataKey="value"
                nameKey="name"
                activeShape={activeShapeRenderer}
                activeIndex={activeIndex}
                onMouseEnter={onPieEnter}
                onMouseLeave={onPieLeave}
              >
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getChartColor(index)} fillOpacity={getOpacity(entry.name)} />
                ))}
              </Pie>
              <Legend
                layout="vertical"
                align="right"
                verticalAlign="middle"
                formatter={legendFormatter}
                onMouseEnter={onLegendMouseEnter}
                onMouseLeave={onLegendMouseLeave}
                wrapperStyle={{
                  fontSize: theme.typography.fontSizeSm,
                  maxHeight: DEFAULT_CHART_CONTENT_HEIGHT,
                  overflowY: 'auto',
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        ) : (
          <OverviewChartEmptyState
            message={
              <FormattedMessage
                defaultMessage="No cost data available"
                description="Message shown when there is no cost data to display"
              />
            }
          />
        )}
      </div>
    </OverviewChartContainer>
  );
};
