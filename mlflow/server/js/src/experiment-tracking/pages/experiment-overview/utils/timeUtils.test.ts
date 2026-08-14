import { describe, it, expect } from '@jest/globals';
import {
  calculateDefaultTimeUnit,
  calculateExpectedDataPoints,
  isTimeUnitValid,
  TimeUnit,
  TIME_UNIT_SECONDS,
  MAX_DATA_POINTS,
} from './timeUtils';

describe('timeUtils', () => {
  describe('calculateDefaultTimeUnit', () => {
    it.each([
      ['startTimeMs is undefined', undefined, Date.now(), TimeUnit.Day],
      ['endTimeMs is undefined', Date.now(), undefined, TimeUnit.Day],
      ['both times are undefined', undefined, undefined, TimeUnit.Day],
    ])('should return Day when %s', (_desc, startTime, endTime, expected) => {
      expect(calculateDefaultTimeUnit(startTime, endTime)).toBe(expected);
    });

    it.each([
      ['30 minutes', 30 * 60 * 1000, TimeUnit.Minute],
      ['exactly 1 hour', 60 * 60 * 1000, TimeUnit.Minute],
      ['1 hour + 1 second', 60 * 60 * 1000 + 1000, TimeUnit.Hour],
      ['12 hours', 12 * 60 * 60 * 1000, TimeUnit.Hour],
      ['exactly 24 hours', 24 * 60 * 60 * 1000, TimeUnit.Hour],
      ['24 hours + 1 second', 24 * 60 * 60 * 1000 + 1000, TimeUnit.Day],
      ['7 days', 7 * 24 * 60 * 60 * 1000, TimeUnit.Day],
      ['30 days', 30 * 24 * 60 * 60 * 1000, TimeUnit.Day],
      ['60 days', 60 * 24 * 60 * 60 * 1000, TimeUnit.Month],
      ['exactly 1 year', 365 * 24 * 60 * 60 * 1000, TimeUnit.Month],
      ['2 years', 2 * 365 * 24 * 60 * 60 * 1000, TimeUnit.Year],
    ])('should return correct unit for %s duration', (_desc, durationMs, expected) => {
      const endTime = Date.now();
      const startTime = endTime - durationMs;
      expect(calculateDefaultTimeUnit(startTime, endTime)).toBe(expected);
    });
  });

  describe('TIME_UNIT_SECONDS', () => {
    it('should have correct values for all time units', () => {
      expect(TIME_UNIT_SECONDS[TimeUnit.Second]).toBe(1);
      expect(TIME_UNIT_SECONDS[TimeUnit.Minute]).toBe(60);
      expect(TIME_UNIT_SECONDS[TimeUnit.Hour]).toBe(3600);
      expect(TIME_UNIT_SECONDS[TimeUnit.Day]).toBe(86400);
      expect(TIME_UNIT_SECONDS[TimeUnit.Month]).toBe(2592000);
      expect(TIME_UNIT_SECONDS[TimeUnit.Year]).toBe(31536000);
    });
  });

  describe('MAX_DATA_POINTS', () => {
    it('should be 1000', () => {
      expect(MAX_DATA_POINTS).toBe(1000);
    });
  });

  describe('calculateExpectedDataPoints', () => {
    it.each([
      ['startTimeMs is undefined', undefined, Date.now()],
      ['endTimeMs is undefined', Date.now(), undefined],
      ['both times are undefined', undefined, undefined],
    ])('should return 0 when %s', (_desc, startTime, endTime) => {
      expect(calculateExpectedDataPoints(startTime, endTime, TimeUnit.Hour)).toBe(0);
    });

    it.each([
      ['1 hour with Second unit', 60 * 60 * 1000, TimeUnit.Second, 3600],
      ['1 hour with Minute unit', 60 * 60 * 1000, TimeUnit.Minute, 60],
      ['7 days with Hour unit', 7 * 24 * 60 * 60 * 1000, TimeUnit.Hour, 168],
      ['7 days with Day unit', 7 * 24 * 60 * 60 * 1000, TimeUnit.Day, 7],
      ['90 seconds with Minute unit (rounds up)', 90 * 1000, TimeUnit.Minute, 2],
    ])('should calculate correct data points for %s', (_desc, durationMs, unit, expected) => {
      const endTime = Date.now();
      const startTime = endTime - durationMs;
      expect(calculateExpectedDataPoints(startTime, endTime, unit)).toBe(expected);
    });
  });

  describe('isTimeUnitValid', () => {
    it('should return true when times are undefined', () => {
      expect(isTimeUnitValid(undefined, Date.now(), TimeUnit.Second)).toBe(true);
      expect(isTimeUnitValid(Date.now(), undefined, TimeUnit.Second)).toBe(true);
    });

    it.each([
      // [description, durationMs, unit, expected]
      ['7 days with Hour (168 points)', 7 * 24 * 60 * 60 * 1000, TimeUnit.Hour, true],
      ['7 days with Second (604800 points)', 7 * 24 * 60 * 60 * 1000, TimeUnit.Second, false],
      ['7 days with Minute (10080 points)', 7 * 24 * 60 * 60 * 1000, TimeUnit.Minute, false],
      ['15 minutes with Second (900 points)', 15 * 60 * 1000, TimeUnit.Second, true],
      ['20 minutes with Second (1200 points)', 20 * 60 * 1000, TimeUnit.Second, false],
      ['exactly 1000 seconds with Second (= MAX)', 1000 * 1000, TimeUnit.Second, true],
      ['1001 seconds with Second (> MAX)', 1001 * 1000, TimeUnit.Second, false],
    ])('should return %s for %s', (_desc, durationMs, unit, expected) => {
      const endTime = Date.now();
      const startTime = endTime - durationMs;
      expect(isTimeUnitValid(startTime, endTime, unit)).toBe(expected);
    });
  });
});
