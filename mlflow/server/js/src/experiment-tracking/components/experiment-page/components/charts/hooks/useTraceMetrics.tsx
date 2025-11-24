import { useQuery } from '@tanstack/react-query';
// import { MlflowService } from '../../../../sdk/MlflowService'; // Uncomment when backend is ready

export interface MetricDataPoint {
  dimensions: Record<string, string>;
  metric_name?: string;
  values: Record<string, string>;
}

interface UseTraceMetricsParams {
  experimentIds: string[];
  timeGranularity: 'HOUR' | 'DAY' | 'MONTH';
  startTimeMs?: number;
  endTimeMs?: number;
  enabled?: boolean;
}

/**
 * Formats a timestamp to a time bucket string based on granularity
 */
const formatTimeBucket = (timestampMs: number, granularity: 'HOUR' | 'DAY' | 'MONTH'): string => {
  const date = new Date(timestampMs);
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  const hour = String(date.getHours()).padStart(2, '0');

  switch (granularity) {
    case 'HOUR':
      return `${year}-${month}-${day} ${hour}:00`;
    case 'DAY':
      return `${year}-${month}-${day}`;
    case 'MONTH':
      return `${year}-${month}`;
    default:
      return `${year}-${month}-${day}`;
  }
};

/**
 * Helper function to generate mock data with a specific range (used internally)
 */
const generateMockDataWithRange = (
  timeGranularity: 'HOUR' | 'DAY' | 'MONTH',
  actualStartTime: number,
  actualEndTime: number,
  intervalMs: number,
): MetricDataPoint[] => {
  const dataPoints: MetricDataPoint[] = [];

  // Round start time down to the beginning of its time bucket
  const startDate = new Date(actualStartTime);
  switch (timeGranularity) {
    case 'HOUR':
      startDate.setMinutes(0, 0, 0);
      startDate.setMilliseconds(0);
      break;
    case 'DAY':
      startDate.setHours(0, 0, 0, 0);
      break;
    case 'MONTH':
      startDate.setDate(1);
      startDate.setHours(0, 0, 0, 0);
      break;
  }

  let currentTimestamp = startDate.getTime();
  const maxIterations = 1000; // Safety limit
  let iterationCount = 0;

  // Generate data points for each time bucket in the range
  while (currentTimestamp <= actualEndTime && iterationCount < maxIterations) {
    iterationCount++;

    // Generate random count between 10-60 for realistic-looking data
    const count = Math.floor(Math.random() * 50) + 10;
    const timeBucket = formatTimeBucket(currentTimestamp, timeGranularity);

    dataPoints.push({
      dimensions: {
        time_bucket: timeBucket,
      },
      metric_name: 'trace_count',
      values: {
        count: count.toString(),
      },
    });

    // Move to next time bucket
    if (timeGranularity === 'MONTH') {
      const nextDate = new Date(currentTimestamp);
      nextDate.setMonth(nextDate.getMonth() + 1);
      currentTimestamp = nextDate.getTime();
    } else {
      currentTimestamp += intervalMs;
    }
  }

  return dataPoints;
};

/**
 * Generates mock trace metrics data for development
 */
const generateMockData = (
  timeGranularity: 'HOUR' | 'DAY' | 'MONTH',
  startTimeMs: number,
  endTimeMs: number,
): MetricDataPoint[] => {
  const now = Date.now();

  // Calculate interval for stepping through time buckets
  let intervalMs: number;
  switch (timeGranularity) {
    case 'HOUR':
      intervalMs = 60 * 60 * 1000; // 1 hour
      break;
    case 'DAY':
      intervalMs = 24 * 60 * 60 * 1000; // 1 day
      break;
    case 'MONTH':
      intervalMs = 0; // Special handling for months (use Date.setMonth)
      break;
  }

  // Use provided time range or default based on granularity
  const actualStartTime = startTimeMs || now - (
    timeGranularity === 'HOUR' ? 24 * 60 * 60 * 1000 : // Last 24 hours
    timeGranularity === 'DAY' ? 7 * 24 * 60 * 60 * 1000 : // Last 7 days
    12 * 30 * 24 * 60 * 60 * 1000 // Last ~12 months
  );
  const actualEndTime = endTimeMs || now;

  // Validate time range
  if (actualStartTime >= actualEndTime) {
    console.warn('Invalid time range: start time must be before end time');
    return [];
  }

  return generateMockDataWithRange(timeGranularity, actualStartTime, actualEndTime, intervalMs);
};

export const useTraceMetrics = ({
  experimentIds,
  timeGranularity,
  startTimeMs,
  endTimeMs,
  enabled = true,
}: UseTraceMetricsParams) => {
  // Serialize experimentIds array to ensure stable queryKey
  const experimentIdsKey = JSON.stringify(experimentIds.sort());
  
  return useQuery<MetricDataPoint[]>({
    queryKey: ['traceMetrics', experimentIdsKey, timeGranularity, startTimeMs, endTimeMs],
    queryFn: async () => {
      // TODO: Replace with real API call once backend is implemented
      // Uncomment the following when backend is ready:
      /*
      if (!experimentIds.length || !startTimeMs || !endTimeMs) {
        return [];
      }
      const response = await MlflowService.queryTraceMetrics({
        experiment_ids: experimentIds,
        view_type: 'TRACES',
        measure: 'trace',
        aggregation_type: ['COUNT'],
        time_granularity: timeGranularity,
        start_time_ms: startTimeMs,
        end_time_ms: endTimeMs,
        max_results: 1000,
      });
      return response.data_points || [];
      */

      // Mock implementation for development
      await new Promise(resolve => setTimeout(resolve, 300));

      // For mock data, use appropriate time ranges for each granularity
      // This ensures we always get a good number of data points
      const now = Date.now();
      let mockStartTime: number;
      let mockEndTime: number;
      
      if (timeGranularity === 'HOUR') {
        // For hourly data, show last 24 hours
        mockEndTime = now;
        mockStartTime = now - 24 * 60 * 60 * 1000;
      } else if (timeGranularity === 'DAY') {
        // For daily data, use the provided range or default to 30 days
        mockEndTime = endTimeMs || now;
        const providedRange = endTimeMs && startTimeMs ? endTimeMs - startTimeMs : 0;
        if (providedRange < 7 * 24 * 60 * 60 * 1000) {
          // If range is less than 7 days, use 30 days
          mockStartTime = mockEndTime - 30 * 24 * 60 * 60 * 1000;
        } else {
          mockStartTime = startTimeMs || mockEndTime - 30 * 24 * 60 * 60 * 1000;
        }
      } else { // MONTH
        // For monthly data, show last 12 months
        mockEndTime = now;
        mockStartTime = now - 12 * 30 * 24 * 60 * 60 * 1000;
      }

      return generateMockData(timeGranularity, mockStartTime, mockEndTime);
    },
    enabled: enabled && experimentIds.length > 0,
    staleTime: 0, // Always refetch when granularity/time range changes
    keepPreviousData: false, // Don't keep previous data to avoid stale data showing
    refetchOnMount: true,
    refetchOnWindowFocus: false,
  });
};

