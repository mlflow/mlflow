/**
 * Common props for overview charts that query trace metrics
 */
export interface OverviewChartProps {
  experimentId: string;
  startTimeMs?: number;
  endTimeMs?: number;
}
