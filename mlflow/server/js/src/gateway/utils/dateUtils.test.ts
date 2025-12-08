import { describe, test, expect } from '@jest/globals';
import { timestampToDate } from './dateUtils';

describe('timestampToDate', () => {
  test('converts Unix timestamp in seconds to Date', () => {
    const timestampInSeconds = 1700000000;
    const result = timestampToDate(timestampInSeconds);

    expect(result.getTime()).toBe(timestampInSeconds * 1000);
    expect(result.getFullYear()).toBe(2023);
  });

  test('handles timestamp already in milliseconds', () => {
    const timestampInMs = 1700000000000;
    const result = timestampToDate(timestampInMs);

    expect(result.getTime()).toBe(timestampInMs);
    expect(result.getFullYear()).toBe(2023);
  });

  test('returns current date for undefined', () => {
    const before = Date.now();
    const result = timestampToDate(undefined);
    const after = Date.now();

    expect(result.getTime()).toBeGreaterThanOrEqual(before);
    expect(result.getTime()).toBeLessThanOrEqual(after);
  });

  test('returns current date for null', () => {
    const before = Date.now();
    const result = timestampToDate(null);
    const after = Date.now();

    expect(result.getTime()).toBeGreaterThanOrEqual(before);
    expect(result.getTime()).toBeLessThanOrEqual(after);
  });

  test('handles boundary case - small seconds timestamp', () => {
    const timestampInSeconds = 1000000001;
    const result = timestampToDate(timestampInSeconds);

    expect(result.getTime()).toBe(timestampInSeconds * 1000);
  });

  test('handles boundary case - large milliseconds timestamp', () => {
    const timestampInMs = 10000000001;
    const result = timestampToDate(timestampInMs);

    expect(result.getTime()).toBe(timestampInMs);
  });
});
