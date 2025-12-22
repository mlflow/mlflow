import { TIME_BUCKET_DIMENSION_KEY, type MetricDataPoint } from '@databricks/web-shared/model-trace-explorer';

/**
 * Format timestamp for x-axis labels based on time interval granularity
 * @param timestampMs - Timestamp in milliseconds
 * @param timeIntervalSeconds - Time interval in seconds used for grouping
 */
export function formatTimestamp(timestampMs: number, timeIntervalSeconds: number): string {
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

