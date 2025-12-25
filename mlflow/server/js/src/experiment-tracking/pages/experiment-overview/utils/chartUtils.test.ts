import { describe, it, expect } from '@jest/globals';
import { formatTimestampForTraceMetrics, getTimestampFromDataPoint } from './chartUtils';
import type { MetricDataPoint } from '@databricks/web-shared/model-trace-explorer';

// Time intervals in seconds
const MINUTE_IN_SECONDS = 60;
const HOUR_IN_SECONDS = 3600;
const DAY_IN_SECONDS = 86400;
const MONTH_IN_SECONDS = 2592000;

describe('chartUtils', () => {
  describe('formatTimestampForTraceMetrics', () => {
    // Use a fixed timestamp for consistent testing: 2025-12-22 10:30:45 UTC
    const testTimestamp = new Date('2025-12-22T10:30:45Z').getTime();

    it('should format timestamp at minute level (timeInterval <= 60s)', () => {
      const result = formatTimestampForTraceMetrics(testTimestamp, MINUTE_IN_SECONDS);
      // Should show time only (hour:minute)
      expect(result).toMatch(/\d{1,2}:\d{2}/);
    });

    it('should format timestamp at hour level (timeInterval <= 3600s)', () => {
      const result = formatTimestampForTraceMetrics(testTimestamp, HOUR_IN_SECONDS);
      // Should show month/day and hour
      expect(result).toMatch(/\d{1,2}\/\d{1,2}/);
    });

    it('should format timestamp at day level (timeInterval <= 86400s)', () => {
      const result = formatTimestampForTraceMetrics(testTimestamp, DAY_IN_SECONDS);
      // Should show month/day
      expect(result).toMatch(/\d{1,2}\/\d{1,2}/);
    });

    it('should format timestamp at month level (timeInterval > 86400s)', () => {
      const result = formatTimestampForTraceMetrics(testTimestamp, MONTH_IN_SECONDS);
      // Should show short month and year (e.g., "Dec '25")
      expect(result).toMatch(/[A-Za-z]+.*\d{2}/);
    });

    it('should handle edge case at exactly 60 seconds (minute level)', () => {
      const result = formatTimestampForTraceMetrics(testTimestamp, 60);
      expect(result).toMatch(/\d{1,2}:\d{2}/);
    });

    it('should handle edge case at exactly 3600 seconds (hour level)', () => {
      const result = formatTimestampForTraceMetrics(testTimestamp, 3600);
      expect(result).toMatch(/\d{1,2}\/\d{1,2}/);
    });

    it('should handle edge case at exactly 86400 seconds (day level)', () => {
      const result = formatTimestampForTraceMetrics(testTimestamp, 86400);
      expect(result).toMatch(/\d{1,2}\/\d{1,2}/);
    });
  });

  describe('getTimestampFromDataPoint', () => {
    it('should extract timestamp from time_bucket dimension', () => {
      const dataPoint: MetricDataPoint = {
        metric_name: 'trace_count',
        dimensions: {
          time_bucket: '2025-12-22T10:00:00Z',
        },
        values: { COUNT: 42 },
      };

      const result = getTimestampFromDataPoint(dataPoint);
      expect(result).toBe(new Date('2025-12-22T10:00:00Z').getTime());
    });

    it('should return 0 when time_bucket is not present', () => {
      const dataPoint: MetricDataPoint = {
        metric_name: 'trace_count',
        dimensions: {
          status: 'OK',
        },
        values: { COUNT: 42 },
      };

      const result = getTimestampFromDataPoint(dataPoint);
      expect(result).toBe(0);
    });

    it('should return 0 when dimensions is empty', () => {
      const dataPoint: MetricDataPoint = {
        metric_name: 'trace_count',
        dimensions: {},
        values: { COUNT: 42 },
      };

      const result = getTimestampFromDataPoint(dataPoint);
      expect(result).toBe(0);
    });

    it('should handle data point with multiple dimensions including time_bucket', () => {
      const dataPoint: MetricDataPoint = {
        metric_name: 'latency',
        dimensions: {
          status: 'OK',
          time_bucket: '2025-12-22T15:30:00Z',
          name: 'my_trace',
        },
        values: { AVG: 150.5, P99: 234.5 },
      };

      const result = getTimestampFromDataPoint(dataPoint);
      expect(result).toBe(new Date('2025-12-22T15:30:00Z').getTime());
    });
  });
});
