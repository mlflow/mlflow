import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

import { catalogToRates, loadCatalogRates } from '../src/modelCatalog';

const CATALOG = {
  schema_version: '1.0',
  models: {
    'claude-sonnet-4-20250514': {
      pricing: {
        input_per_million_tokens: 3,
        output_per_million_tokens: 15,
        cache_read_per_million_tokens: 0.3,
        cache_write_per_million_tokens: 3.75,
      },
    },
    'no-pricing-model': {},
  },
};

const EXPECTED_RATES = {
  'claude-sonnet-4-20250514': { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
};

function okFetch(body: unknown = CATALOG): jest.Mock {
  return jest.fn().mockResolvedValue({ ok: true, text: async () => JSON.stringify(body) });
}

function failFetch(): jest.Mock {
  return jest.fn().mockRejectedValue(new Error('network down'));
}

function tempCacheDir(): string {
  return mkdtempSync(join(tmpdir(), 'mlflow-catalog-test-'));
}

describe('catalogToRates', () => {
  it('converts pricing entries and skips models without base pricing', () => {
    expect(catalogToRates(CATALOG)).toEqual(EXPECTED_RATES);
  });

  it('omits cache rates when absent', () => {
    const rates = catalogToRates({
      models: {
        m: { pricing: { input_per_million_tokens: 1, output_per_million_tokens: 2 } },
      },
    });
    expect(rates).toEqual({ m: { input: 1, output: 2 } });
  });
});

describe('loadCatalogRates', () => {
  it('fetches, returns rates, and writes the cache file', async () => {
    const fetchImpl = okFetch();
    const cacheDir = tempCacheDir();

    const rates = await loadCatalogRates({
      baseUri: 'https://example.com/base/',
      cacheDir,
      fetchImpl,
    });

    expect(rates).toEqual(EXPECTED_RATES);
    expect(fetchImpl).toHaveBeenCalledWith(
      'https://example.com/base/anthropic.json',
      expect.anything(),
    );
    const cached = JSON.parse(readFileSync(join(cacheDir, 'anthropic.json'), 'utf8'));
    expect(cached.rates).toEqual(EXPECTED_RATES);
  });

  it('serves from the cache within the TTL without fetching', async () => {
    const cacheDir = tempCacheDir();
    await loadCatalogRates({ baseUri: 'https://example.com', cacheDir, fetchImpl: okFetch() });

    const fetchImpl = failFetch();
    const rates = await loadCatalogRates({ baseUri: 'https://example.com', cacheDir, fetchImpl });

    expect(rates).toEqual(EXPECTED_RATES);
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it('caches a failed fetch so it is not retried within the TTL', async () => {
    const cacheDir = tempCacheDir();
    const first = await loadCatalogRates({
      baseUri: 'https://example.com',
      cacheDir,
      fetchImpl: failFetch(),
    });
    expect(first).toBeNull();

    const fetchImpl = okFetch();
    const second = await loadCatalogRates({ baseUri: 'https://example.com', cacheDir, fetchImpl });
    expect(second).toBeNull();
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it('bypasses the cache when ttlSeconds is 0', async () => {
    const cacheDir = tempCacheDir();
    await loadCatalogRates({ baseUri: 'https://example.com', cacheDir, fetchImpl: okFetch() });

    const fetchImpl = okFetch();
    await loadCatalogRates({ baseUri: 'https://example.com', cacheDir, fetchImpl, ttlSeconds: 0 });
    expect(fetchImpl).toHaveBeenCalled();
  });

  it('returns null without fetching when the base URI is empty', async () => {
    const fetchImpl = okFetch();
    const rates = await loadCatalogRates({ baseUri: '', cacheDir: tempCacheDir(), fetchImpl });
    expect(rates).toBeNull();
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it('returns null for a non-OK response', async () => {
    const fetchImpl = jest.fn().mockResolvedValue({ ok: false, text: async () => 'not found' });
    const rates = await loadCatalogRates({
      baseUri: 'https://example.com',
      cacheDir: tempCacheDir(),
      fetchImpl,
    });
    expect(rates).toBeNull();
  });

  it('returns null for a catalog with no usable rates', async () => {
    const rates = await loadCatalogRates({
      baseUri: 'https://example.com',
      cacheDir: tempCacheDir(),
      fetchImpl: okFetch({ models: { m: {} } }),
    });
    expect(rates).toBeNull();
  });

  it('reads from a file:// mirror', async () => {
    const mirrorDir = tempCacheDir();
    writeFileSync(join(mirrorDir, 'anthropic.json'), JSON.stringify(CATALOG));

    const fetchImpl = okFetch();
    const rates = await loadCatalogRates({
      baseUri: pathToFileURL(mirrorDir).href,
      cacheDir: tempCacheDir(),
      fetchImpl,
    });

    expect(rates).toEqual(EXPECTED_RATES);
    expect(fetchImpl).not.toHaveBeenCalled();
  });
});
