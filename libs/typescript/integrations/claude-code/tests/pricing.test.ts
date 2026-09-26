import { calculateCost, setModelRates, sumCosts } from '../src/pricing';

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

  it('prices cached tokens at the catalog cache rates', () => {
    // claude-sonnet-4: cacheRead=0.3/MTok, cacheWrite=3.75/MTok
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

  it('prices each model individually, not by family prefix', () => {
    // Opus 4.5+ is cheaper (5/25) than Opus 4/4.1 (15/75); a prefix match on
    // "claude-opus-4" would overcharge it 3x.
    const opus48 = calculateCost('claude-opus-4-8', {
      input_tokens: 1_000_000,
      output_tokens: 1_000_000,
    });
    expect(opus48!.input_cost).toBeCloseTo(5, 6);
    expect(opus48!.output_cost).toBeCloseTo(25, 6);
  });

  it('falls back to the undated family alias for unknown dated snapshots', () => {
    const cost = calculateCost('claude-opus-4-8-20991231', {
      input_tokens: 1_000_000,
      output_tokens: 0,
    });
    expect(cost!.input_cost).toBeCloseTo(5, 6);
  });

  it('returns null for unknown models', () => {
    expect(calculateCost('gpt-4o', { input_tokens: 100, output_tokens: 100 })).toBeNull();
    // Models absent from the catalog (e.g. deprecated ones) get no cost,
    // matching the Python pipeline reading the same catalog.
    expect(
      calculateCost('claude-3-5-sonnet-20241022', { input_tokens: 100, output_tokens: 100 }),
    ).toBeNull();
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

describe('setModelRates', () => {
  afterEach(() => setModelRates(null));

  it('overrides the bundled rates until reset', () => {
    setModelRates({ 'custom-model': { input: 100, output: 200 } });

    const cost = calculateCost('custom-model', { input_tokens: 1_000_000, output_tokens: 0 });
    expect(cost!.input_cost).toBeCloseTo(100, 6);
    // Bundled models are replaced wholesale, matching Python's remote-first load.
    expect(
      calculateCost('claude-sonnet-4-20250514', { input_tokens: 100, output_tokens: 0 }),
    ).toBeNull();

    setModelRates(null);
    expect(
      calculateCost('claude-sonnet-4-20250514', { input_tokens: 100, output_tokens: 0 }),
    ).not.toBeNull();
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
