import { describe, expect, test } from '@jest/globals';
import { normalizeComparisonRunUuids } from './evalRunComparisonData';

describe('comparison facts', () => {
  test('normalizes duplicate selections while preserving the baseline and order', () => {
    expect(normalizeComparisonRunUuids('a', ['b', 'a', '', 'b', 'c'])).toEqual(['a', 'b', 'c']);
  });
});
