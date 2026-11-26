import { describe, test, expect } from '@jest/globals';
import { formatDuration } from './budgetFormatUtils';

describe('formatDuration', () => {
  test.each([
    [1, 'MINUTES' as const, '1 Minute'],
    [15, 'MINUTES' as const, '15 Minutes'],
    [1, 'HOURS' as const, 'Hourly'],
    [6, 'HOURS' as const, '6 Hours'],
    [1, 'DAYS' as const, 'Daily'],
    [2, 'WEEKS' as const, '2 Weeks'],
    [1, 'MONTHS' as const, 'Monthly'],
  ])('formats %d %s as %s', (value, unit, expected) => {
    expect(formatDuration(value, unit)).toBe(expected);
  });
});
