import { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import { TIME_BUCKET_DIMENSION_KEY, type MetricDataPoint } from '@databricks/web-shared/model-trace-explorer';
import { useDesignSystemTheme, type DesignSystemThemeInterface } from '@databricks/design-system';

/**
 * Props for the active shape renderer in pie charts.
 */
export interface ActiveShapeProps {
  cx: number;
  cy: number;
  innerRadius: number;
  outerRadius: number;
  startAngle: number;
  endAngle: number;
  fill: string;
  name: string;
  value: number;
  percentage: number;
  midAngle: number;
}

/**
 * Geometry values for rendering a pie chart active shape with external labels.
 */
export interface PieActiveShapeGeometry {
  /** Line start point X (just outside pie) */
  sx: number;
  /** Line start point Y */
  sy: number;
  /** Line bend point X */
  mx: number;
  /** Line bend point Y */
  my: number;
  /** Line end point X (horizontal offset) */
  ex: number;
  /** Line end point Y */
  ey: number;
  /** Text anchor for labels ('start' or 'end') */
  textAnchor: 'start' | 'end';
  /** Cosine of midAngle (for label offset direction) */
  cos: number;
}

/**
 * Calculates geometry for rendering a pie chart active shape with external label lines.
 * Used to position connecting lines and labels outside the pie slice.
 *
 * @param props - Active shape props from Recharts
 * @param theme - Design system theme for spacing values
 * @returns Geometry values for rendering the active shape
 */
export function calculatePieActiveShapeGeometry(
  props: ActiveShapeProps,
  theme: DesignSystemThemeInterface['theme'],
): PieActiveShapeGeometry {
  const { cx, cy, outerRadius, midAngle } = props;
  const RADIAN = Math.PI / 180;

  // Calculate direction from center based on slice's midpoint angle
  const sin = Math.sin(-RADIAN * midAngle);
  const cos = Math.cos(-RADIAN * midAngle);

  // Line start point (just outside the pie slice)
  const sx = cx + (outerRadius + theme.spacing.sm) * cos;
  const sy = cy + (outerRadius + theme.spacing.sm) * sin;

  // Line bend point (further out, creates an elbow in the line)
  const mx = cx + (outerRadius + theme.spacing.md) * cos;
  const my = cy + (outerRadius + theme.spacing.md) * sin;

  // Line end point (horizontal offset from bend, direction based on left/right side)
  const ex = mx + (cos >= 0 ? 1 : -1) * theme.spacing.md;
  const ey = my;
  const textAnchor = cos >= 0 ? 'start' : 'end';

  return { sx, sy, mx, my, ex, ey, textAnchor, cos };
}

/**
 * Custom hook for managing legend highlight state in charts.
 * Returns state and handlers for highlighting chart series on legend hover.
 *
 * @param defaultOpacity - The opacity when no item is hovered (default: 1)
 * @param dimmedOpacity - The opacity for non-hovered items (default: 0.2)
 */
export function useLegendHighlight(defaultOpacity = 1, dimmedOpacity = 0.2) {
  const [hoveredItem, setHoveredItem] = useState<string | null>(null);

  const getOpacity = useCallback(
    (itemKey: string) => {
      if (hoveredItem === null) return defaultOpacity;
      return hoveredItem === itemKey ? defaultOpacity : dimmedOpacity;
    },
    [hoveredItem, defaultOpacity, dimmedOpacity],
  );

  const handleLegendMouseEnter = useCallback((data: { value: string }) => {
    setHoveredItem(data.value);
  }, []);

  const handleLegendMouseLeave = useCallback(() => {
    setHoveredItem(null);
  }, []);

  return {
    hoveredItem,
    getOpacity,
    handleLegendMouseEnter,
    handleLegendMouseLeave,
  };
}

/**
 * Creates a Map from data points, extracting timestamp as key and a value using the provided extractor.
 * This is useful for looking up values by timestamp when filling in missing time buckets.
 *
 * @param dataPoints - Array of metric data points
 * @param valueExtractor - Function to extract the value from each data point
 */
export function useTimestampValueMap<T>(
  dataPoints: MetricDataPoint[],
  valueExtractor: (dp: MetricDataPoint) => T,
): Map<number, T> {
  return useMemo(() => {
    const map = new Map<number, T>();
    for (const dp of dataPoints) {
      const timeBucket = dp.dimensions?.[TIME_BUCKET_DIMENSION_KEY];
      if (timeBucket) {
        const ts = new Date(timeBucket).getTime();
        map.set(ts, valueExtractor(dp));
      }
    }
    return map;
  }, [dataPoints, valueExtractor]);
}

/**
 * Format timestamp for trace metrics charts x-axis labels based on time interval granularity
 * @param timestampMs - Timestamp in milliseconds
 * @param timeIntervalSeconds - Time interval in seconds used for grouping
 */
export function formatTimestampForTraceMetrics(timestampMs: number, timeIntervalSeconds: number): string {
  const date = new Date(timestampMs);

  // Build format options based on time interval granularity
  let options: Intl.DateTimeFormatOptions;
  if (timeIntervalSeconds <= 60) {
    // Minute level - show time only (e.g., "10:30")
    options = { hour: '2-digit', minute: '2-digit' };
  } else if (timeIntervalSeconds <= 3600) {
    // Hour level - show date and hour (e.g., "12/22, 10 AM")
    options = { month: 'numeric', day: 'numeric', hour: '2-digit' };
  } else if (timeIntervalSeconds <= 86400) {
    // Day level - show month/day (e.g., "12/22")
    options = { month: 'numeric', day: 'numeric' };
  } else {
    // Month level - show short month and year (e.g., "Dec '25")
    options = { month: 'short', year: '2-digit' };
  }

  return date.toLocaleString([], options);
}

/**
 * Extract timestamp from time_bucket dimension of a metric data point
 * @param dp - Metric data point containing dimensions
 * @returns Timestamp in milliseconds, or 0 if not found
 */
export function getTimestampFromDataPoint(dp: MetricDataPoint): number {
  const timeBucket = dp.dimensions?.[TIME_BUCKET_DIMENSION_KEY];
  if (timeBucket) {
    return new Date(timeBucket).getTime();
  }
  return 0;
}

/**
 * Generate all time bucket timestamps within a range
 * @param startTimeMs - Start of time range in milliseconds
 * @param endTimeMs - End of time range in milliseconds
 * @param timeIntervalSeconds - Time interval in seconds for each bucket
 * @returns Array of timestamps (in ms) for each bucket, aligned to interval boundaries
 */
export function generateTimeBuckets(
  startTimeMs: number | undefined,
  endTimeMs: number | undefined,
  timeIntervalSeconds: number,
): number[] {
  if (!startTimeMs || !endTimeMs || timeIntervalSeconds <= 0) {
    return [];
  }

  const intervalMs = timeIntervalSeconds * 1000;
  const buckets: number[] = [];

  // Align start time to the interval boundary (floor)
  const alignedStart = Math.floor(startTimeMs / intervalMs) * intervalMs;

  for (let ts = alignedStart; ts <= endTimeMs; ts += intervalMs) {
    buckets.push(ts);
  }

  return buckets;
}

/**
 * Default dot style configuration for line charts.
 * Creates small, solid dots that match the line color.
 *
 * @param color - The fill color for the dot (should match line stroke)
 * @returns Dot props object for Recharts Line component
 */
export function getLineDotStyle(color: string) {
  return { r: 2, fill: color, strokeWidth: 0 };
}

/**
 * Format a large number with K/M suffix for human-readable display
 * @param count - Number to format
 * @returns Formatted string (e.g., "1.50M", "15.00K", "1.50K", "500")
 */
export function formatCount(count: number): string {
  if (count >= 1_000_000) {
    return `${(count / 1_000_000).toFixed(2)}M`;
  }
  if (count >= 1_000) {
    return `${(count / 1_000).toFixed(2)}K`;
  }
  return count.toLocaleString();
}

/**
 * Formats latency in milliseconds to a human-readable string
 * @param ms - Latency in milliseconds
 */
export function formatLatency(ms: number): string {
  if (ms < 1000) {
    return `${ms.toFixed(2)}ms`;
  }
  return `${(ms / 1000).toFixed(2)}s`;
}

/**
 * Custom hook providing a color palette for charts.
 * Returns a memoized color array and a getter function for consistent coloring by index.
 * Used for tool charts, assessment charts, and other visualizations needing distinct colors.
 */
export function useChartColors() {
  const { theme } = useDesignSystemTheme();

  const chartColors = useMemo(
    () => [
      theme.colors.blue500,
      theme.colors.green500,
      theme.colors.yellow500,
      theme.colors.red500,
      theme.colors.blue300,
      theme.colors.green300,
      theme.colors.yellow300,
      theme.colors.red300,
      theme.colors.blue400,
      theme.colors.green400,
      theme.colors.yellow400,
      theme.colors.red400,
      theme.colors.blue600,
      theme.colors.green600,
      theme.colors.yellow600,
      theme.colors.red600,
      theme.colors.blue700,
      theme.colors.green700,
      theme.colors.yellow700,
      theme.colors.red700,
    ],
    [theme],
  );

  const getChartColor = useCallback((index: number): string => chartColors[index % chartColors.length], [chartColors]);

  return { chartColors, getChartColor };
}

/**
 * State and handlers for chart zoom functionality.
 * Implements click-and-drag range selection to zoom into a portion of the chart.
 */
export interface ChartZoomState<T> {
  /** Currently displayed (potentially zoomed) data */
  zoomedData: T[];
  /** Whether the chart is currently zoomed in */
  isZoomed: boolean;
  /** Left boundary of current selection (index or value) */
  refAreaLeft: string | number | null;
  /** Right boundary of current selection (index or value) */
  refAreaRight: string | number | null;
  /** Handler for mouse down event - starts selection */
  handleMouseDown: (e: { activeLabel?: string }) => void;
  /** Handler for mouse move event - updates selection */
  handleMouseMove: (e: { activeLabel?: string }) => void;
  /** Handler for mouse up event - completes zoom */
  handleMouseUp: () => void;
  /** Reset zoom to show all data */
  zoomOut: () => void;
}

/**
 * Custom hook for implementing zoom functionality in Recharts.
 * Allows users to click and drag to select a range, then zooms to that range.
 *
 * @param data - The full dataset for the chart
 * @param labelKey - The key used for x-axis labels (e.g., 'name')
 * @returns Zoom state and handlers
 */
export function useChartZoom<T>(data: T[], labelKey: keyof T): ChartZoomState<T> {
  const [zoomedData, setZoomedData] = useState<T[]>(data);
  const [refAreaLeft, setRefAreaLeft] = useState<string | number | null>(null);
  const [refAreaRight, setRefAreaRight] = useState<string | number | null>(null);
  const [isSelecting, setIsSelecting] = useState(false);
  const [isZoomed, setIsZoomed] = useState(false);

  // Track previous data to detect when source data changes
  const prevDataRef = useRef<T[]>(data);

  // Reset zoom when source data changes (e.g., time range changed)
  useEffect(() => {
    // Check if data reference or length changed (indicates new data, not just same data)
    const dataChanged = prevDataRef.current !== data;
    if (dataChanged) {
      setZoomedData(data);
      setIsZoomed(false);
      setRefAreaLeft(null);
      setRefAreaRight(null);
      prevDataRef.current = data;
    }
  }, [data]);

  const handleMouseDown = useCallback((e: { activeLabel?: string }) => {
    if (e.activeLabel) {
      setRefAreaLeft(e.activeLabel);
      setRefAreaRight(e.activeLabel);
      setIsSelecting(true);
    }
  }, []);

  const handleMouseMove = useCallback(
    (e: { activeLabel?: string }) => {
      if (isSelecting && e.activeLabel) {
        setRefAreaRight(e.activeLabel);
      }
    },
    [isSelecting],
  );

  const handleMouseUp = useCallback(() => {
    if (!isSelecting || refAreaLeft === null || refAreaRight === null) {
      setIsSelecting(false);
      return;
    }

    // Find indices of the selected range
    let leftIndex = data.findIndex((d) => d[labelKey] === refAreaLeft);
    let rightIndex = data.findIndex((d) => d[labelKey] === refAreaRight);

    // Ensure left is before right
    if (leftIndex > rightIndex) {
      [leftIndex, rightIndex] = [rightIndex, leftIndex];
    }

    // Only zoom if we have a valid range with at least 2 points
    if (leftIndex >= 0 && rightIndex >= 0 && rightIndex - leftIndex >= 1) {
      setZoomedData(data.slice(leftIndex, rightIndex + 1));
      setIsZoomed(true);
    }

    // Reset selection state
    setRefAreaLeft(null);
    setRefAreaRight(null);
    setIsSelecting(false);
  }, [isSelecting, refAreaLeft, refAreaRight, data, labelKey]);

  const zoomOut = useCallback(() => {
    setZoomedData(data);
    setIsZoomed(false);
    setRefAreaLeft(null);
    setRefAreaRight(null);
  }, [data]);

  return {
    zoomedData,
    isZoomed,
    refAreaLeft,
    refAreaRight,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    zoomOut,
  };
}
