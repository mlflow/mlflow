import { calculateCost, setModelRates, sumCosts } from '../src/pricing';

describe('calculateCost', () => {
  it('prices input and output tokens at the model rate', () => {
    // gpt-4o: input 2.5/MTok, output 10/MTok
    const cost = calculateCost('gpt-4o', {
      input_tokens: 1_000_000,
      output_tokens: 1_000_000,
      total_tokens: 2_000_000,
    });
    expect(cost).not.toBeNull();
    expect(cost!.input_cost).toBeCloseTo(2.5, 6);
    expect(cost!.output_cost).toBeCloseTo(10, 6);
    expect(cost!.total_cost).toBeCloseTo(12.5, 6);
  });

  it('prices the cached subset of input at the cache-read rate', () => {
    // gpt-4o: input 2.5/MTok, cacheRead 1.25/MTok. OpenAI reports
    // cached_input_tokens as a subset of input_tokens.
    const cost = calculateCost('gpt-4o', {
      input_tokens: 1000,
      output_tokens: 1000,
      total_tokens: 2000,
      cached_input_tokens: 400,
    });
    // 600 uncached * 2.5 + 400 cached * 1.25 (per MTok)
    expect(cost!.input_cost).toBeCloseTo(0.002, 9);
    expect(cost!.output_cost).toBeCloseTo(0.01, 9);
    expect(cost!.total_cost).toBeCloseTo(0.012, 9);
  });

  it('caps cached tokens at the reported input tokens', () => {
    // A cached count larger than input_tokens must not produce negative
    // uncached input; the whole input is priced at the cache-read rate.
    const cost = calculateCost('gpt-4o', {
      input_tokens: 1000,
      output_tokens: 0,
      total_tokens: 1000,
      cached_input_tokens: 5000,
    });
    expect(cost!.input_cost).toBeCloseTo(1000 * 1.25e-6, 12);
  });

  it('prices each model individually, not by family prefix', () => {
    // gpt-5 (1.25/10) is far cheaper than gpt-4 (30/60); a prefix match would
    // mis-price them.
    const gpt5 = calculateCost('gpt-5', {
      input_tokens: 1_000_000,
      output_tokens: 1_000_000,
      total_tokens: 2_000_000,
    });
    expect(gpt5!.input_cost).toBeCloseTo(1.25, 6);
    expect(gpt5!.output_cost).toBeCloseTo(10, 6);

    const gpt4 = calculateCost('gpt-4', {
      input_tokens: 1_000_000,
      output_tokens: 0,
      total_tokens: 1_000_000,
    });
    expect(gpt4!.input_cost).toBeCloseTo(30, 6);
  });

  it('falls back to the undated family alias for unknown dated snapshots', () => {
    // gpt-5-2099-12-31 is not in the catalog; it should price as gpt-5.
    const cost = calculateCost('gpt-5-2099-12-31', {
      input_tokens: 1_000_000,
      output_tokens: 0,
      total_tokens: 1_000_000,
    });
    expect(cost!.input_cost).toBeCloseTo(1.25, 6);
  });

  it('returns null for unknown models', () => {
    // Codex-family model names are not yet in the published catalog, so they
    // get no cost, matching the Python pipeline reading the same catalog.
    expect(
      calculateCost('gpt-5-codex', { input_tokens: 100, output_tokens: 100, total_tokens: 200 }),
    ).toBeNull();
    // A different provider's model is also absent from openai.json.
    expect(
      calculateCost('claude-opus-4-8', {
        input_tokens: 100,
        output_tokens: 100,
        total_tokens: 200,
      }),
    ).toBeNull();
  });

  it('returns null when there is no billable base usage', () => {
    expect(
      calculateCost('gpt-4o', { input_tokens: 0, output_tokens: 0, total_tokens: 0 }),
    ).toBeNull();
  });

  it('returns null for missing model or usage', () => {
    expect(
      calculateCost(undefined, { input_tokens: 10, output_tokens: 10, total_tokens: 20 }),
    ).toBeNull();
    expect(calculateCost('gpt-4o', undefined)).toBeNull();
  });
});

describe('setModelRates', () => {
  afterEach(() => setModelRates(null));

  it('overrides the bundled rates until reset', () => {
    setModelRates({ 'custom-model': { input: 100, output: 200 } });

    const cost = calculateCost('custom-model', {
      input_tokens: 1_000_000,
      output_tokens: 0,
      total_tokens: 1_000_000,
    });
    expect(cost!.input_cost).toBeCloseTo(100, 6);
    // Bundled models are replaced wholesale, matching Python's remote-first load.
    expect(
      calculateCost('gpt-4o', { input_tokens: 100, output_tokens: 0, total_tokens: 100 }),
    ).toBeNull();

    setModelRates(null);
    expect(
      calculateCost('gpt-4o', { input_tokens: 100, output_tokens: 0, total_tokens: 100 }),
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
