/**
 * GENERATED FILE - DO NOT EDIT BY HAND.
 *
 * Snapshot of Anthropic per-model pricing from MLflow's bundled model catalog
 * (mlflow/utils/model_catalog/anthropic.json). Regenerate with:
 *
 *   npm run sync:pricing
 *
 * tests/anthropicPricing.test.ts fails when this file drifts from the catalog.
 */

/** Per-million-token USD rates for one Claude model. */
export interface AnthropicModelRate {
  /** USD per million regular (non-cached) input tokens. */
  input: number;
  /** USD per million output tokens. */
  output: number;
  /** USD per million cache-read input tokens. */
  cacheRead?: number;
  /** USD per million cache-write (cache creation) input tokens. */
  cacheWrite?: number;
}

export const ANTHROPIC_MODEL_RATES: Readonly<Record<string, AnthropicModelRate>> = {
  'claude-3-7-sonnet-20250219': { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
  'claude-3-haiku-20240307': { input: 0.25, output: 1.25, cacheRead: 0.03, cacheWrite: 0.3 },
  'claude-3-opus-20240229': { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 },
  'claude-4-opus-20250514': { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 },
  'claude-4-sonnet-20250514': { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
  'claude-fable-5': { input: 10, output: 50, cacheRead: 1, cacheWrite: 12.5 },
  'claude-haiku-4-5': { input: 1, output: 5, cacheRead: 0.1, cacheWrite: 1.25 },
  'claude-haiku-4-5-20251001': { input: 1, output: 5, cacheRead: 0.1, cacheWrite: 1.25 },
  'claude-opus-4-1': { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 },
  'claude-opus-4-1-20250805': { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 },
  'claude-opus-4-20250514': { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 },
  'claude-opus-4-5': { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
  'claude-opus-4-5-20251101': { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
  'claude-opus-4-6': { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
  'claude-opus-4-6-20260205': { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
  'claude-opus-4-7': { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
  'claude-opus-4-7-20260416': { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
  'claude-opus-4-8': { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
  'claude-sonnet-4-20250514': { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
  'claude-sonnet-4-5': { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
  'claude-sonnet-4-5-20250929': { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
  'claude-sonnet-4-6': { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
  'claude-sonnet-5': { input: 2, output: 10, cacheRead: 0.2, cacheWrite: 2.5 },
};
