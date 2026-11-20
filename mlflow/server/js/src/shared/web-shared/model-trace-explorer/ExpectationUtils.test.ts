import { describe, it, expect } from '@jest/globals';
import { getParsedExpectationValue } from './ExpectationUtils';
import type { Expectation } from './ModelTrace.types';

describe('getParsedExpectationValue', () => {
  it.each([[false], [true], [0], [''], ['[1, 2]'], [null]])('preserves the direct value %p', (value) => {
    expect(getParsedExpectationValue({ value })).toBe(value);
  });

  it.each<[string, unknown]>([
    ['42', 42],
    ['false', false],
    ['0', 0],
    ['""', ''],
    ['[1, 2]', [1, 2]],
    ['{"label":"positive"}', { label: 'positive' }],
    ['"text"', 'text'],
    ['null', null],
    ['unfinished {', 'unfinished {'],
  ])('decodes serialized values or preserves the original text for %s', (value, expected) => {
    expect(getParsedExpectationValue({ serialized_value: { value, serialization_format: 'JSON_FORMAT' } })).toEqual(
      expected,
    );
  });

  it('preserves the trace extractor fallback for a missing serialized value', () => {
    expect(getParsedExpectationValue({} as Expectation)).toBeNull();
  });
});
