import React from 'react';

/**
 * Mock implementations for recharts components.
 */
export const mockRechartsComponents = {
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="responsive-container">{children}</div>
  ),
  LineChart: ({ children, data }: { children: React.ReactNode; data: any[] }) => (
    <div data-testid="line-chart" data-count={data?.length || 0}>
      {children}
    </div>
  ),
  BarChart: ({ children, data }: { children: React.ReactNode; data: any[] }) => (
    <div data-testid="bar-chart" data-count={data?.length || 0}>
      {children}
    </div>
  ),
  ComposedChart: ({ children, data }: { children: React.ReactNode; data: any[] }) => (
    <div data-testid="composed-chart" data-count={data?.length || 0}>
      {children}
    </div>
  ),
  AreaChart: ({ children, data }: { children: React.ReactNode; data: any[] }) => (
    <div data-testid="area-chart" data-count={data?.length || 0}>
      {children}
    </div>
  ),
  Line: ({ name }: { name?: string }) => <div data-testid={name ? `line-${name}` : 'line'} />,
  Bar: ({ name }: { name?: string }) => <div data-testid={name ? `bar-${name}` : 'bar'} />,
  Area: ({ name }: { name?: string }) => <div data-testid={name ? `area-${name}` : 'area'} />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
  ReferenceLine: ({ label }: { label?: { value: string } }) => (
    <div data-testid="reference-line" data-label={label?.value} />
  ),
};
