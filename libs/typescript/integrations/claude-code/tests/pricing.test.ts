import { calculateCost, sumCosts } from '../src/pricing';

describe('calculateCost', () => {
  it('prices input and output tokens at the model rate', () => {
    const cost = calculateCost('claude-sonnet-4-20250514', {
      input_tokens: 1000,
      output_tokens: 1000,
    });
    expect(cost).not.toBeNull();
    expect(cost!.input_cost).toBeCloseTo(0.003, 9);
    expect(cost!.output_cost).toBeCloseTo(0.015, 9);
    expect(cost!.total_cost).toBeCloseTo(0.018, 9);
  });

  it('prices cache read at 0.1x and cache write at 1.25x the input rate', () => {
    const cost = calculateCost('claude-sonnet-4-20250514', {
      input_tokens: 10,
      output_tokens: 25,
      cache_read_input_tokens: 40,
      cache_creation_input_tokens: 100,
    });
    expect(cost!.input_cost).toBeCloseTo(0.000417, 9);
    expect(cost!.output_cost).toBeCloseTo(0.000375, 9);
    expect(cost!.total_cost).toBeCloseTo(0.000792, 9);
  });

  it('applies opus and haiku rates', () => {
    const opus = calculateCost('claude-opus-4-1-20250805', {
      input_tokens: 1_000_000,
      output_tokens: 0,
    });
    expect(opus!.input_cost).toBeCloseTo(15, 6);
    const haiku = calculateCost('claude-haiku-4-5-20251001', {
      input_tokens: 0,
      output_tokens: 1_000_000,
    });
    expect(haiku!.output_cost).toBeCloseTo(5, 6);
  });

  it('returns null for unknown models', () => {
    expect(calculateCost('gpt-4o', { input_tokens: 100, output_tokens: 100 })).toBeNull();
  });

  it('returns null when there is no billable base usage', () => {
    expect(
      calculateCost('claude-sonnet-4-20250514', { input_tokens: 0, output_tokens: 0 }),
    ).toBeNull();
  });

  it('returns null for missing model or usage', () => {
    expect(calculateCost(undefined, { input_tokens: 10, output_tokens: 10 })).toBeNull();
    expect(calculateCost('claude-sonnet-4-20250514', undefined)).toBeNull();
  });
});

describe('sumCosts', () => {
  it('sums component costs', () => {
    expect(
      sumCosts([
        { input_cost: 1, output_cost: 2, total_cost: 3 },
        { input_cost: 0.5, output_cost: 0.5, total_cost: 1 },
      ]),
    ).toEqual({ input_cost: 1.5, output_cost: 2.5, total_cost: 4 });
  });

  it('returns null for an empty list', () => {
    expect(sumCosts([])).toBeNull();
  });
});
