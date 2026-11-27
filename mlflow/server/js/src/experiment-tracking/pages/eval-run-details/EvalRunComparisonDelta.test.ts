import { describe, expect, test } from '@jest/globals';
import { createIntl } from 'react-intl';
import { formatEvalRunComparisonNumericValue, formatEvalRunComparisonPercentChange } from './EvalRunComparisonDelta';

const intl = createIntl({ locale: 'en', messages: {} });

const NUMERIC_VALUE_CASES: {
  value: number;
  signDisplay?: Parameters<typeof formatEvalRunComparisonNumericValue>[2];
  expected: string;
}[] = [
  { value: 0.6523, expected: '0.65' },
  { value: 0.7, expected: '0.70' },
  { value: 12.346, expected: '12.35' },
  { value: 1, expected: '1' },
  { value: 0.999, expected: '1' },
  { value: 0.05, signDisplay: 'exceptZero', expected: '+0.05' },
  { value: -0.2, signDisplay: 'exceptZero', expected: '-0.20' },
  { value: -5, signDisplay: 'exceptZero', expected: '-5' },
];

describe('formatEvalRunComparisonNumericValue', () => {
  // @ui-test-skill-generated
  test.each(NUMERIC_VALUE_CASES)('formats $value as $expected', ({ value, signDisplay, expected }) => {
    expect(formatEvalRunComparisonNumericValue(intl, value, signDisplay)).toBe(expected);
  });
});

describe('formatEvalRunComparisonPercentChange', () => {
  // @ui-test-skill-generated
  test.each([
    { percent: 84, expected: '+84%' },
    { percent: 100 / 3, expected: '+33.33%' },
    { percent: 7.6923, expected: '+7.69%' },
    { percent: -100, expected: '-100%' },
  ])('formats $percent as $expected', ({ percent, expected }) => {
    expect(formatEvalRunComparisonPercentChange(intl, percent)).toBe(expected);
  });
});
