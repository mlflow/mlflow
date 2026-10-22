import type { ComponentProps, ReactNode } from 'react';
import { useDesignSystemTheme } from '@databricks/design-system';
import { Bar, BarChart, ResponsiveContainer, Tooltip } from 'recharts';

type RechartsBarChartProps = ComponentProps<typeof BarChart>;
type RechartsTooltipProps = ComponentProps<typeof Tooltip>;

export interface TraceRequestsBarChartProps {
  bars?: ReactNode;
  children?: ReactNode;
  data: NonNullable<RechartsBarChartProps['data']>;
  height: number;
  margin?: RechartsBarChartProps['margin'];
  onMouseDown?: RechartsBarChartProps['onMouseDown'];
  onMouseLeave?: RechartsBarChartProps['onMouseLeave'];
  onMouseMove?: RechartsBarChartProps['onMouseMove'];
  onMouseUp?: RechartsBarChartProps['onMouseUp'];
  tooltipContent?: RechartsTooltipProps['content'];
}

export const TraceRequestsBarChart = ({
  bars,
  children,
  data,
  height,
  margin,
  onMouseDown,
  onMouseLeave,
  onMouseMove,
  onMouseUp,
  tooltipContent,
}: TraceRequestsBarChartProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ height, userSelect: 'none' }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={margin}
          onMouseDown={onMouseDown}
          onMouseLeave={onMouseLeave}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
        >
          <Tooltip
            content={tooltipContent ?? (() => null)}
            cursor={{ fill: theme.colors.actionTertiaryBackgroundHover }}
          />
          {bars ?? <Bar dataKey="count" fill={theme.colors.blue400} radius={[4, 4, 0, 0]} />}
          {children}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
