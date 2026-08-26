import { readFileSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { calculateCostByModelAndTokenUsage } from '../src/core/utils/model_pricing';
import {
  MODEL_PRICING_CATALOG,
  type ModelPricingCatalog,
} from '../src/core/utils/model_pricing_catalog';

describe('bundled model pricing', () => {
  it('calculates base and cached token cost using the Python catalog rates', () => {
    const cost = calculateCostByModelAndTokenUsage(
      'gpt-5-mini',
      {
        input_tokens: 110,
        output_tokens: 20,
        cache_read_input_tokens: 10,
      },
      'openai.responses',
    );

    expect(cost?.input_cost).toBeCloseTo(0.00002525, 12);
    expect(cost?.output_cost).toBeCloseTo(0.00004, 12);
    expect(cost?.total_cost).toBeCloseTo(0.00006525, 12);
  });

  it('falls back to the regular input rate when cache-write pricing is absent', () => {
    const cost = calculateCostByModelAndTokenUsage(
      'openai/gpt-5-mini',
      {
        input_tokens: 10,
        output_tokens: 0,
        cache_creation_input_tokens: 4,
      },
      'openai',
    );

    expect(cost?.input_cost).toBeCloseTo(0.0000025, 12);
    expect(cost?.total_cost).toBeCloseTo(0.0000025, 12);
  });

  it('returns undefined for unknown models and empty usage', () => {
    expect(
      calculateCostByModelAndTokenUsage('unknown-model', {
        input_tokens: 1,
        output_tokens: 1,
      }),
    ).toBeUndefined();
    expect(
      calculateCostByModelAndTokenUsage('gpt-5-mini', {
        input_tokens: 0,
        output_tokens: 0,
      }),
    ).toBeUndefined();
  });

  it('matches the pricing fields in the Python model catalog', () => {
    const catalogDir = resolve(__dirname, '../../../../mlflow/utils/model_catalog');
    const expected: Record<string, Record<string, Record<string, number>>> = {};

    for (const filename of readdirSync(catalogDir)
      .filter((name) => name.endsWith('.json'))
      .sort()) {
      const provider = filename.slice(0, -'.json'.length);
      const source = JSON.parse(readFileSync(resolve(catalogDir, filename), 'utf8')) as {
        models?: Record<string, { pricing?: Record<string, unknown> }>;
      };
      const models: Record<string, Record<string, number>> = {};
      for (const [model, entry] of Object.entries(source.models ?? {})) {
        const pricing = entry.pricing;
        if (!pricing) {
          continue;
        }
        const rate: Record<string, number> = {};
        for (const [sourceKey, targetKey] of [
          ['input_per_million_tokens', 'inputPerMillion'],
          ['output_per_million_tokens', 'outputPerMillion'],
          ['cache_read_per_million_tokens', 'cacheReadPerMillion'],
          ['cache_write_per_million_tokens', 'cacheWritePerMillion'],
        ]) {
          const value = pricing[sourceKey];
          if (typeof value === 'number' && Number.isFinite(value)) {
            rate[targetKey] = value;
          }
        }
        if (Object.keys(rate).length > 0) {
          models[model] = rate;
        }
      }
      if (Object.keys(models).length > 0) {
        expected[provider] = models;
      }
    }

    expect(MODEL_PRICING_CATALOG).toEqual(expected satisfies ModelPricingCatalog);
  });
});
