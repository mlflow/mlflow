/**
 * Common props for overview charts that query trace metrics
 */
export interface OverviewChartProps {
  experimentId: string;
  startTimeMs?: number;
  endTimeMs?: number;
  /** Time interval in seconds for grouping metrics by time bucket */
  timeIntervalSeconds: number;
  /** Pre-computed array of timestamp (ms) for all time buckets in the range */
  timeBuckets: number[];
}
