/**
 * Remote model-catalog lookup, mirroring Python's `mlflow/utils/providers.py`.
 *
 * Rates are fetched from the published catalog (a GitHub Release asset updated
 * weekly by CI) so pricing stays current between npm releases. Python caches
 * fetches in memory with a TTL; the Stop hook is a short-lived process that runs
 * after every assistant turn, so the cache lives on the filesystem instead.
 * Like Python, a failed fetch is also cached for the TTL (as a `rates: null`
 * marker) so offline machines stall at most once per TTL window, and callers
 * fall back to the bundled snapshot (`anthropicPricing.ts`) when this returns
 * null.
 *
 * Same semantics as Python: `MLFLOW_MODEL_CATALOG_URI` overrides the base URI
 * (`file://` mirrors supported for air-gapped environments; empty string
 * disables remote lookup), and `MLFLOW_MODEL_CATALOG_CACHE_TTL` sets the cache
 * TTL in seconds (0 disables caching).
 */
import { mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { AnthropicModelRate } from './anthropicPricing.js';

export const MODEL_CATALOG_URI_ENV = 'MLFLOW_MODEL_CATALOG_URI';
export const MODEL_CATALOG_CACHE_TTL_ENV = 'MLFLOW_MODEL_CATALOG_CACHE_TTL';

const DEFAULT_CATALOG_URI =
  'https://github.com/mlflow/mlflow/releases/download/model-catalog%2Flatest';
const DEFAULT_CACHE_TTL_SECONDS = 86400;
const FETCH_TIMEOUT_MS = 5000;
const CACHE_SCHEMA_VERSION = 1;

interface CatalogPricing {
  input_per_million_tokens?: number;
  output_per_million_tokens?: number;
  cache_read_per_million_tokens?: number;
  cache_write_per_million_tokens?: number;
}

interface CatalogFile {
  models?: Record<string, { pricing?: CatalogPricing }>;
}

interface CacheFile {
  schemaVersion: number;
  fetchedAt: string;
  /** null records a failed fetch so it is not retried within the TTL. */
  rates: Record<string, AnthropicModelRate> | null;
}

export interface LoadCatalogOptions {
  /** Base URI override (default: MLFLOW_MODEL_CATALOG_URI or the GitHub Release). */
  baseUri?: string;
  /** Cache directory override (default: ~/.mlflow/model_catalog_cache). */
  cacheDir?: string;
  /** Cache TTL override in seconds (default: MLFLOW_MODEL_CATALOG_CACHE_TTL or 86400). */
  ttlSeconds?: number;
  /** fetch override for tests. */
  fetchImpl?: typeof fetch;
}

/** Convert a catalog file to the per-model rate map used by calculateCost. */
export function catalogToRates(catalog: CatalogFile): Record<string, AnthropicModelRate> {
  const rates: Record<string, AnthropicModelRate> = {};
  for (const [name, entry] of Object.entries(catalog.models ?? {})) {
    const pricing = entry.pricing;
    if (pricing?.input_per_million_tokens == null || pricing?.output_per_million_tokens == null) {
      continue;
    }
    rates[name] = {
      input: pricing.input_per_million_tokens,
      output: pricing.output_per_million_tokens,
      ...(pricing.cache_read_per_million_tokens != null && {
        cacheRead: pricing.cache_read_per_million_tokens,
      }),
      ...(pricing.cache_write_per_million_tokens != null && {
        cacheWrite: pricing.cache_write_per_million_tokens,
      }),
    };
  }
  return rates;
}

function envTtlSeconds(): number {
  const raw = process.env[MODEL_CATALOG_CACHE_TTL_ENV];
  const parsed = raw == null || raw.trim() === '' ? NaN : Number(raw);
  return Number.isFinite(parsed) ? parsed : DEFAULT_CACHE_TTL_SECONDS;
}

/** Read the cache file; undefined = miss (absent, stale, or unreadable). */
function readCache(
  cachePath: string,
  ttlSeconds: number,
): Record<string, AnthropicModelRate> | null | undefined {
  try {
    if (Date.now() - statSync(cachePath).mtimeMs > ttlSeconds * 1000) {
      return undefined;
    }
    const cached = JSON.parse(readFileSync(cachePath, 'utf8')) as CacheFile;
    if (cached.schemaVersion !== CACHE_SCHEMA_VERSION) {
      return undefined;
    }
    if (cached.rates == null) {
      return null;
    }
    return typeof cached.rates === 'object' ? cached.rates : undefined;
  } catch {
    return undefined;
  }
}

function writeCache(cachePath: string, rates: Record<string, AnthropicModelRate> | null): void {
  // Best-effort: an unwritable cache dir just means refetching next time.
  try {
    mkdirSync(join(cachePath, '..'), { recursive: true });
    const cache: CacheFile = {
      schemaVersion: CACHE_SCHEMA_VERSION,
      fetchedAt: new Date().toISOString(),
      rates,
    };
    writeFileSync(cachePath, JSON.stringify(cache));
  } catch {
    // ignore
  }
}

async function fetchRates(
  baseUri: string,
  fetchImpl: typeof fetch,
): Promise<Record<string, AnthropicModelRate> | null> {
  try {
    const url = `${baseUri.replace(/\/+$/, '')}/anthropic.json`;
    let raw: string;
    if (url.startsWith('file:')) {
      raw = readFileSync(fileURLToPath(new URL(url)), 'utf8');
    } else {
      const res = await fetchImpl(url, { signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) });
      if (!res.ok) {
        return null;
      }
      raw = await res.text();
    }
    const rates = catalogToRates(JSON.parse(raw) as CatalogFile);
    return Object.keys(rates).length ? rates : null;
  } catch {
    return null;
  }
}

/**
 * Load model rates from the remote catalog, using the filesystem cache when
 * fresh. Returns null when remote lookup is disabled, fails, or yields no
 * usable rates — callers fall back to the bundled snapshot.
 */
export async function loadCatalogRates(
  options: LoadCatalogOptions = {},
): Promise<Record<string, AnthropicModelRate> | null> {
  const baseUri = options.baseUri ?? process.env[MODEL_CATALOG_URI_ENV] ?? DEFAULT_CATALOG_URI;
  if (!baseUri.trim()) {
    return null;
  }

  const ttlSeconds = options.ttlSeconds ?? envTtlSeconds();
  const cachePath = join(
    options.cacheDir ?? join(homedir(), '.mlflow', 'model_catalog_cache'),
    'anthropic.json',
  );

  if (ttlSeconds > 0) {
    const cached = readCache(cachePath, ttlSeconds);
    if (cached !== undefined) {
      return cached;
    }
  }

  const rates = await fetchRates(baseUri, options.fetchImpl ?? fetch);
  writeCache(cachePath, rates);
  return rates;
}
