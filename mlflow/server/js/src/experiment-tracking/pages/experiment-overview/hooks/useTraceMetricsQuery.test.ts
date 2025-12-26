import { describe, it, expect } from '@jest/globals';
import { calculateTimeInterval } from './useTraceMetricsQuery';

// Time intervals in seconds
const MINUTE_IN_SECONDS = 60;
const HOUR_IN_SECONDS = 3600;
const DAY_IN_SECONDS = 86400;
const MONTH_IN_SECONDS = 2592000;

describe('useTraceMetricsQuery', () => {
  describe('calculateTimeInterval', () => {
    it('should return DAY_IN_SECONDS when startTimeMs is undefined', () => {
      const result = calculateTimeInterval(undefined, Date.now());
      expect(result).toBe(DAY_IN_SECONDS);
    });

    it('should return DAY_IN_SECONDS when endTimeMs is undefined', () => {
      const result = calculateTimeInterval(Date.now(), undefined);
      expect(result).toBe(DAY_IN_SECONDS);
    });

    it('should return DAY_IN_SECONDS when both times are undefined', () => {
      const result = calculateTimeInterval(undefined, undefined);
      expect(result).toBe(DAY_IN_SECONDS);
    });

    it('should return MINUTE_IN_SECONDS for duration <= 1 hour', () => {
      const endTime = Date.now();
      const startTime = endTime - 30 * 60 * 1000; // 30 minutes ago

      const result = calculateTimeInterval(startTime, endTime);
      expect(result).toBe(MINUTE_IN_SECONDS);
    });

    it('should return MINUTE_IN_SECONDS for exactly 1 hour duration', () => {
      const endTime = Date.now();
      const startTime = endTime - 1 * 60 * 60 * 1000; // Exactly 1 hour ago

      const result = calculateTimeInterval(startTime, endTime);
      expect(result).toBe(MINUTE_IN_SECONDS);
    });

    it('should return HOUR_IN_SECONDS for duration > 1 hour and <= 24 hours', () => {
      const endTime = Date.now();
      const startTime = endTime - 12 * 60 * 60 * 1000; // 12 hours ago

      const result = calculateTimeInterval(startTime, endTime);
      expect(result).toBe(HOUR_IN_SECONDS);
    });

    it('should return HOUR_IN_SECONDS for exactly 24 hours duration', () => {
      const endTime = Date.now();
      const startTime = endTime - 24 * 60 * 60 * 1000; // Exactly 24 hours ago

      const result = calculateTimeInterval(startTime, endTime);
      expect(result).toBe(HOUR_IN_SECONDS);
    });

    it('should return DAY_IN_SECONDS for duration > 24 hours and <= 1 month', () => {
      const endTime = Date.now();
      const startTime = endTime - 7 * 24 * 60 * 60 * 1000; // 7 days ago

      const result = calculateTimeInterval(startTime, endTime);
      expect(result).toBe(DAY_IN_SECONDS);
    });

    it('should return DAY_IN_SECONDS for exactly 30 days duration', () => {
      const endTime = Date.now();
      const startTime = endTime - 30 * 24 * 60 * 60 * 1000; // 30 days ago (= MONTH_IN_SECONDS)

      const result = calculateTimeInterval(startTime, endTime);
      expect(result).toBe(DAY_IN_SECONDS);
    });

    it('should return MONTH_IN_SECONDS for duration > 1 month', () => {
      const endTime = Date.now();
      const startTime = endTime - 60 * 24 * 60 * 60 * 1000; // 60 days ago

      const result = calculateTimeInterval(startTime, endTime);
      expect(result).toBe(MONTH_IN_SECONDS);
    });

    it('should return MONTH_IN_SECONDS for very long durations (1 year)', () => {
      const endTime = Date.now();
      const startTime = endTime - 365 * 24 * 60 * 60 * 1000; // 1 year ago

      const result = calculateTimeInterval(startTime, endTime);
      expect(result).toBe(MONTH_IN_SECONDS);
    });

    it('should handle edge case just over 1 hour', () => {
      const endTime = Date.now();
      const startTime = endTime - (1 * 60 * 60 * 1000 + 1000); // 1 hour + 1 second ago

      const result = calculateTimeInterval(startTime, endTime);
      expect(result).toBe(HOUR_IN_SECONDS);
    });

    it('should handle edge case just over 24 hours', () => {
      const endTime = Date.now();
      const startTime = endTime - (24 * 60 * 60 * 1000 + 1000); // 24 hours + 1 second ago

      const result = calculateTimeInterval(startTime, endTime);
      expect(result).toBe(DAY_IN_SECONDS);
    });
  });
});
