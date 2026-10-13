import React, { useCallback, useMemo, useState } from 'react';
import { useDesignSystemTheme, PieChartIcon, type DesignSystemThemeInterface } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { PieChart, Pie, ResponsiveContainer, Legend, Sector, Tooltip } from 'recharts';
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
import { useChartColors, useLegendHighlight, type ActiveShapeProps } from '../utils/chartUtils';

// Pie chart sizing constants
const PIE_INNER_RADIUS = 50;
const PIE_OUTER_RADIUS = 70;
const PIE_PADDING_ANGLE = 2;

/**
 * Renders the active shape for a pie slice with outer arc, connecting line, and labels.
 */
const renderActiveShape = (props: ActiveShapeProps, theme: DesignSystemThemeInterface['theme']) => {
  const { cx, cy, innerRadius, outerRadius, startAngle, endAngle, fill, name, value, percentage } = props;

  // Label is always centered above the pie — no overflow issues regardless of slice position
  const labelY = cy - outerRadius - theme.spacing.lg - 4;

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
      {/* Model name */}
      <text
        x={cx}
        y={labelY}
        textAnchor="middle"
        fill={theme.colors.textPrimary}
        fontSize={theme.typography.fontSizeSm}
        fontWeight={500}
      >
        {name}
      </text>
      {/* Cost and percentage */}
      <text
        x={cx}
        y={labelY + theme.spacing.md}
        textAnchor="middle"
        fill={theme.colors.textSecondary}
        fontSize={theme.typography.fontSizeSm}
      >
        {formatCostUSD(value)}, {percentage.toFixed(2)}%
      </text>
    </g>
  );
};

interface ShapeProps extends ActiveShapeProps {
  isActive: boolean;
  index: number;
  fillOpacity?: number;
}

/**
 * Creates a shape renderer that shows active styling for:
 * - Pie slice hover (via Tooltip's isActive)
 * - Legend hover (via legendActiveIndex)
 */
const createShapeRenderer =
  (theme: DesignSystemThemeInterface['theme'], legendActiveIndex: number | undefined) =>
  (props: ShapeProps): React.ReactElement => {
    const isActive = props.isActive || legendActiveIndex === props.index;

    if (isActive) {
      return renderActiveShape(props, theme);
    }

    return (
      <Sector
        cx={props.cx}
        cy={props.cy}
        innerRadius={props.innerRadius}
        outerRadius={props.outerRadius}
        startAngle={props.startAngle}
        endAngle={props.endAngle}
        fill={props.fill}
        fillOpacity={props.fillOpacity}
      />
    );
  };

export const TraceCostBreakdownChart: React.FC = () => {
  const { theme } = useDesignSystemTheme();
  const { dimension, setDimension } = useTraceCostDimension();
  const { chartData, totalCost, isLoading, error, hasData } = useTraceCostBreakdownChartData(dimension);
  const { getChartColor } = useChartColors();
  const { getOpacity, handleLegendMouseEnter, handleLegendMouseLeave } = useLegendHighlight();
  const [legendActiveIndex, setLegendActiveIndex] = useState<number | undefined>(undefined);

  const shapeRenderer = useMemo(() => createShapeRenderer(theme, legendActiveIndex), [theme, legendActiveIndex]);

  // Add fill colors and opacity directly to data
  const coloredChartData = useMemo(
    () =>
      chartData.map((entry, index) => ({
        ...entry,
        fill: getChartColor(index),
        fillOpacity: getOpacity(entry.name),
      })),
    [chartData, getChartColor, getOpacity],
  );

  const onLegendMouseEnter = useCallback(
    (data: { value: string | undefined }) => {
      handleLegendMouseEnter(data);
      const index = chartData.findIndex((entry) => entry.name === data.value);
      if (index !== -1) {
        setLegendActiveIndex(index);
      }
    },
    [handleLegendMouseEnter, chartData],
  );

  const onLegendMouseLeave = useCallback(() => {
    handleLegendMouseLeave();
    setLegendActiveIndex(undefined);
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
                data={coloredChartData}
                cx="50%"
                cy="60%"
                innerRadius={PIE_INNER_RADIUS}
                outerRadius={PIE_OUTER_RADIUS}
                paddingAngle={PIE_PADDING_ANGLE}
                dataKey="value"
                nameKey="name"
                shape={shapeRenderer}
              />
              {/* Tooltip enables internal active state tracking for pie hover (recharts 3.x) */}
              <Tooltip content={() => null} cursor={false} />
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
