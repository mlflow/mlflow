import React from 'react';

// Mock recharts components to avoid rendering issues in tests
const BarChartDataContext = React.createContext<unknown[]>([]);
const LineChartDataContext = React.createContext<unknown[]>([]);

export const ResponsiveContainer = ({ children }: { children?: React.ReactNode }) => (
  <div data-testid="responsive-container">{children}</div>
);

export const PieChart = ({ children }: { children?: React.ReactNode }) => <div data-testid="pie-chart">{children}</div>;

export const Pie = ({ children, data }: { children?: React.ReactNode; data?: unknown[] }) => (
  <div data-testid="pie" data-count={data?.length || 0}>
    {children}
  </div>
);

export const Cell = ({
  fillOpacity,
  onMouseEnter,
  onMouseLeave,
  role,
  'aria-label': ariaLabel,
  'data-eval-runs-chart-bar': isEvalRunsChartBar,
  'data-run-uuid': runUuid,
  'data-assessment': assessment,
}: {
  fillOpacity?: number;
  onMouseEnter?: React.MouseEventHandler<HTMLDivElement>;
  onMouseLeave?: React.MouseEventHandler<HTMLDivElement>;
  role?: string;
  'aria-label'?: string;
  'data-eval-runs-chart-bar'?: boolean;
  'data-run-uuid'?: string;
  'data-assessment'?: string;
}) => (
  <div
    data-testid="cell"
    data-eval-runs-chart-bar={isEvalRunsChartBar ? '' : undefined}
    data-run-uuid={runUuid}
    data-assessment={assessment}
    data-fill-opacity={fillOpacity}
    onMouseEnter={onMouseEnter}
    onMouseLeave={onMouseLeave}
    role={role}
    aria-label={ariaLabel}
  />
);
export const Sector = () => <div data-testid="sector" />;

export const LineChart = ({
  children,
  data,
  onDoubleClick,
  onMouseDown,
  onMouseLeave,
  onMouseMove,
  onMouseUp,
}: {
  children?: React.ReactNode;
  data?: Array<{ axisKey?: string; runUuid?: string }>;
  onDoubleClick?: (state: unknown, event: React.MouseEvent<HTMLDivElement>) => void;
  onMouseDown?: (state: unknown, event: React.MouseEvent<HTMLDivElement>) => void;
  onMouseLeave?: (state: unknown, event: React.MouseEvent<HTMLDivElement>) => void;
  onMouseMove?: (state: unknown, event: React.MouseEvent<HTMLDivElement>) => void;
  onMouseUp?: (state: unknown, event: React.MouseEvent<HTMLDivElement>) => void;
}) => {
  const chartData = data ?? [];
  const getMouseState = (index: number) => ({
    activeLabel: chartData[index]?.axisKey ?? chartData[index]?.runUuid,
    activeTooltipIndex: index,
  });
  const dragEndIndex = Math.min(1, chartData.length - 1);
  const getDragState = (event: React.MouseEvent<HTMLDivElement>) =>
    getMouseState(event.clientX >= 50 ? dragEndIndex : 0);
  return (
    <LineChartDataContext.Provider value={chartData}>
      <div
        data-testid="line-chart"
        data-count={chartData.length}
        onDoubleClick={(event) => onDoubleClick?.(getMouseState(0), event)}
        onMouseDown={(event) => onMouseDown?.(getMouseState(0), event)}
        onMouseMove={(event) => onMouseMove?.(getDragState(event), event)}
        onPointerDown={(event) => onMouseDown?.(getMouseState(0), event)}
        onPointerMove={(event) => onMouseMove?.(getMouseState(dragEndIndex), event)}
        onPointerUp={(event) => onMouseUp?.(getMouseState(dragEndIndex), event)}
        onMouseUp={(event) => onMouseUp?.(getDragState(event), event)}
        onMouseLeave={(event) => onMouseLeave?.(getMouseState(0), event)}
      >
        {children}
      </div>
    </LineChartDataContext.Provider>
  );
};

export const BarChart = ({
  children,
  data,
  onDoubleClick,
  onMouseDown,
  onMouseLeave,
  onMouseMove,
  onMouseUp,
}: {
  children?: React.ReactNode;
  data?: Array<{ axisKey?: string; name?: string; runName?: string }>;
  onDoubleClick?: (state: unknown, event: React.MouseEvent<HTMLDivElement>) => void;
  onMouseDown?: (state: unknown, event: React.MouseEvent<HTMLDivElement>) => void;
  onMouseLeave?: (state: unknown, event: React.MouseEvent<HTMLDivElement>) => void;
  onMouseMove?: (state: unknown, event: React.MouseEvent<HTMLDivElement>) => void;
  onMouseUp?: (state: unknown, event: React.MouseEvent<HTMLDivElement>) => void;
}) => {
  const chartData = data ?? [];
  const getMouseState = (index: number) => ({
    activeLabel: chartData[index]?.axisKey,
    activeTooltipIndex: index,
  });
  const dragEndIndex = Math.min(1, chartData.length - 1);
  const getDragState = (event: React.MouseEvent<HTMLDivElement>) =>
    getMouseState(event.clientX >= 50 ? dragEndIndex : 0);
  return (
    <BarChartDataContext.Provider value={chartData}>
      <div
        data-testid="bar-chart"
        data-count={chartData.length}
        data-labels={data?.map((d) => d.name).join(',')}
        onDoubleClick={(event) => onDoubleClick?.(getMouseState(0), event)}
        onMouseDown={(event) => onMouseDown?.(getMouseState(0), event)}
        onMouseMove={(event) => onMouseMove?.(getDragState(event), event)}
        onPointerDown={(event) => onMouseDown?.(getMouseState(0), event)}
        onPointerMove={(event) => onMouseMove?.(getMouseState(dragEndIndex), event)}
        onPointerUp={(event) => onMouseUp?.(getMouseState(dragEndIndex), event)}
        onMouseUp={(event) => onMouseUp?.(getDragState(event), event)}
        onMouseLeave={(event) => onMouseLeave?.(getMouseState(0), event)}
      >
        {children}
      </div>
    </BarChartDataContext.Provider>
  );
};

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

export const Line = ({
  name,
  dataKey,
  connectNulls,
  stroke,
}: {
  name?: string;
  dataKey?: string;
  connectNulls?: boolean;
  stroke?: string;
}) => {
  const data = React.useContext(LineChartDataContext);
  const values = data.map((entry) =>
    (typeof dataKey === 'string' ? dataKey : undefined)?.split('.').reduce<unknown>((value, key) => {
      if (typeof value !== 'object' || value === null) {
        return undefined;
      }
      return (value as Record<string, unknown>)[key];
    }, entry),
  );
  return (
    <div
      data-testid={name ? `line-${name}` : 'line'}
      data-key={typeof dataKey === 'string' ? dataKey : undefined}
      data-values={JSON.stringify(values)}
      data-connect-nulls={String(connectNulls)}
      data-stroke={stroke}
    />
  );
};

export const Bar = ({
  fill,
  children,
  name,
  fillOpacity,
  onMouseEnter,
  onMouseLeave,
}: {
  children?: React.ReactNode;
  fill?: string;
  name?: string;
  fillOpacity?: number;
  onMouseEnter?: (data: { payload: unknown }, index: number, event: React.MouseEvent<HTMLDivElement>) => void;
  onMouseLeave?: (data: { payload: unknown }, index: number, event: React.MouseEvent<HTMLDivElement>) => void;
}) => {
  const data = React.useContext(BarChartDataContext);
  const hasCellChildren = React.Children.toArray(children).some(
    (child) => React.isValidElement(child) && child.type === Cell,
  );
  const eventData = { payload: data[0] };
  return (
    <div
      data-testid={name ? `bar-${name}` : 'bar'}
      data-fill={fill}
      data-fill-opacity={fillOpacity}
      onMouseEnter={!hasCellChildren && onMouseEnter ? (event) => onMouseEnter(eventData, 0, event) : undefined}
      onMouseLeave={!hasCellChildren && onMouseLeave ? (event) => onMouseLeave(eventData, 0, event) : undefined}
    >
      {React.Children.map(children, (child, index) => {
        if (!React.isValidElement<React.ComponentProps<typeof Cell>>(child)) {
          return child;
        }
        const cellEventData = { payload: data[index] };
        return React.cloneElement(child, {
          onMouseEnter: onMouseEnter
            ? (event: React.MouseEvent<HTMLDivElement>) => onMouseEnter(cellEventData, index, event)
            : child.props.onMouseEnter,
          onMouseLeave: onMouseLeave
            ? (event: React.MouseEvent<HTMLDivElement>) => onMouseLeave(cellEventData, index, event)
            : child.props.onMouseLeave,
        });
      })}
    </div>
  );
};

export const Area = ({ name, dataKey }: { name?: string; dataKey?: string }) => (
  <div data-testid={`area-${dataKey}`} data-name={name} />
);

export const XAxis = ({
  hide,
  dataKey,
  tick,
  label,
}: {
  hide?: boolean;
  dataKey?: string;
  tick?: unknown;
  label?: { value: string };
}) => (
  <div data-testid="x-axis" data-hide={String(Boolean(hide) || tick === false)} data-key={dataKey}>
    {label?.value}
  </div>
);

export const YAxis = () => <div data-testid="y-axis" />;

export const Tooltip = ({
  labelFormatter,
}: {
  labelFormatter?: (label: string, payload: { payload: unknown }[]) => React.ReactNode;
}) => {
  const data = React.useContext(LineChartDataContext);
  return (
    <div data-testid="tooltip">
      {labelFormatter?.(
        'Hovered run',
        data.slice(0, 1).map((payload) => ({ payload })),
      )}
    </div>
  );
};

export const Legend = () => <div data-testid="legend" />;

export const ReferenceArea = ({ x1, x2 }: { x1?: string | number; x2?: string | number }) => (
  <div data-testid="reference-area" data-x1={x1} data-x2={x2} />
);

export const ReferenceLine = ({ label }: { label?: { value?: string } }) => (
  <div data-testid="reference-line" data-label={label?.value} />
);
