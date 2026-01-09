import { useState, useCallback, useMemo } from 'react';
import { TIME_BUCKET_DIMENSION_KEY, type MetricDataPoint } from '@databricks/web-shared/model-trace-explorer';
import { useDesignSystemTheme } from '@databricks/design-system';

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
 * Custom hook providing a color palette for tool charts.
 * Returns a memoized color array and a getter function for consistent tool coloring.
 */
export function useToolColors() {
  const { theme } = useDesignSystemTheme();

  const toolColors = useMemo(
    () => [
      theme.colors.blue500,
      theme.colors.green500,
      theme.colors.yellow500,
      theme.colors.red500,
      theme.colors.blue300,
      theme.colors.green300,
      theme.colors.yellow300,
      theme.colors.red300,
    ],
    [theme],
  );

  const getToolColor = useCallback((index: number): string => toolColors[index % toolColors.length], [toolColors]);

  return { toolColors, getToolColor };
}
