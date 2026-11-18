import { describe, expect, it, jest } from '@jest/globals';

import { formatCostUSD, formatEstimatedCostUSD } from './CostUtils';

describe('formatCostUSD', () => {
  it.each([0, -0.012345, 0.000001, 1.234567, 1000000, NaN, Infinity])(
    'preserves currency formatting for %s',
    (cost) => {
      for (const maximumFractionDigits of [2, 4, 6]) {
        const expected = new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: 'USD',
          minimumFractionDigits: 2,
          maximumFractionDigits,
        }).format(cost);
        expect(formatCostUSD(cost, maximumFractionDigits)).toBe(expected);
        expect(formatEstimatedCostUSD(cost, maximumFractionDigits)).toBe(`≈${expected}`);
      }
    },
  );

  it('does not construct formatters again for repeated metrics at the same precision', () => {
    const spy = jest.spyOn(Intl, 'NumberFormat');
    try {
      for (let index = 0; index < 1000; index++) {
        formatCostUSD(index / 1000, 5);
        formatEstimatedCostUSD(index / 1000, 7);
      }
      expect(spy).toHaveBeenCalledTimes(2);
    } finally {
      spy.mockRestore();
    }
  });

  it('keeps rejecting invalid precision', () => {
    expect(() => formatCostUSD(1, -1)).toThrow(RangeError);
  });
});
