import React from 'react';

// Mock recharts components to avoid rendering issues in tests
export const ResponsiveContainer = ({ children }: { children?: React.ReactNode }) => (
  <div data-testid="responsive-container">{children}</div>
);

export const LineChart = ({ children, data }: { children?: React.ReactNode; data?: unknown[] }) => (
  <div data-testid="line-chart" data-count={data?.length || 0}>
    {children}
  </div>
);

export const BarChart = ({ children, data }: { children?: React.ReactNode; data?: Array<{ name?: string }> }) => (
  <div data-testid="bar-chart" data-count={data?.length || 0} data-labels={data?.map((d) => d.name).join(',')}>
    {children}
  </div>
);

export const ComposedChart = ({ children, data }: { children?: React.ReactNode; data?: unknown[] }) => (
  <div data-testid="composed-chart" data-count={data?.length || 0}>
    {children}
  </div>
);

export const AreaChart = ({ children, data }: { children?: React.ReactNode; data?: unknown[] }) => (
  <div data-testid="area-chart" data-count={data?.length || 0}>
    {children}
  </div>
);

export const Line = ({ name }: { name?: string }) => <div data-testid={name ? `line-${name}` : 'line'} />;

export const Bar = ({ name }: { name?: string }) => <div data-testid={name ? `bar-${name}` : 'bar'} />;

export const Area = ({ name, dataKey }: { name?: string; dataKey?: string }) => (
  <div data-testid={`area-${dataKey}`} data-name={name} />
);

export const XAxis = () => <div data-testid="x-axis" />;

export const YAxis = () => <div data-testid="y-axis" />;

export const Tooltip = () => <div data-testid="tooltip" />;

export const Legend = () => <div data-testid="legend" />;

export const ReferenceLine = ({ label }: { label?: { value?: string } }) => (
  <div data-testid="reference-line" data-label={label?.value} />
);
