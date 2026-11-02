import { describe, expect, test } from '@jest/globals';
import { formatEvalRunsAxisValue, formatEvalRunsNumericValue } from './ExperimentEvaluationRunsNumberFormat';

describe('formatEvalRunsNumericValue', () => {
  test('rounds to at most four decimal places and trims trailing zeroes', () => {
    expect(formatEvalRunsNumericValue(0.123456789)).toBe('0.1235');
    expect(formatEvalRunsNumericValue(-0.123456789)).toBe('-0.1235');
    expect(formatEvalRunsNumericValue(12.5)).toBe('12.5');
    expect(formatEvalRunsNumericValue(12)).toBe('12');
  });

  test('keeps large numbers out of exponential notation and bounds tiny numbers', () => {
    expect(formatEvalRunsNumericValue(1.23456789e25)).toBe('12345678900000000000000000');
    expect(formatEvalRunsNumericValue(0.0000123456789)).toBe('0');
  });

  test('keeps non-finite values explicit for callers that choose to display them', () => {
    expect(formatEvalRunsNumericValue(Number.NaN)).toBe('NaN');
    expect(formatEvalRunsNumericValue(Number.POSITIVE_INFINITY)).toBe('Infinity');
  });
});

describe('formatEvalRunsAxisValue', () => {
  test('rounds axis ticks to at most two decimal places and trims trailing zeroes', () => {
    expect(formatEvalRunsAxisValue(233.4833)).toBe('233.48');
    expect(formatEvalRunsAxisValue(-0.125)).toBe('-0.13');
    expect(formatEvalRunsAxisValue(12.5)).toBe('12.5');
    expect(formatEvalRunsAxisValue(50)).toBe('50');
  });
});
