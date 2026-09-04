import { describe, expect, test } from '@jest/globals';
import { formatTraceDuration } from './formatTraceDuration';

describe('formatTraceDuration', () => {
  test.each<{ input: string; expected: string }>([
    { input: '0.002s', expected: '2ms' },
    { input: '0.5s', expected: '500ms' },
    { input: '32.583s', expected: '32.6s' },
    { input: '90s', expected: '1.5min' },
    { input: '3600s', expected: '1.0h' },
    { input: '90000s', expected: '1.0d' },
    // Non-second units are still parsed defensively.
    { input: '250ms', expected: '250ms' },
    { input: '1500ms', expected: '1.5s' },
  ])('formats $input as $expected', ({ input, expected }) => {
    expect(formatTraceDuration(input)).toBe(expected);
  });

  test('returns null for an unparseable value so the caller can fall back to the raw string', () => {
    expect(formatTraceDuration('not-a-duration')).toBeNull();
    expect(formatTraceDuration('')).toBeNull();
    expect(formatTraceDuration('12')).toBeNull();
  });
});
