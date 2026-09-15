#!/usr/bin/env node
/**
 * Regenerate src/openaiPricing.ts from MLflow's bundled model catalog
 * (mlflow/utils/model_catalog/openai.json), so the plugin prices OpenAI
 * models from the same data as MLflow's Python cost pipeline.
 *
 * Usage: npm run sync:pricing
 *
 * tests/openaiPricing.test.ts fails when the generated file drifts from
 * the catalog.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const packageDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = join(packageDir, '../../../..');
const catalogPath = join(repoRoot, 'mlflow/utils/model_catalog/openai.json');
const outputPath = join(packageDir, 'src/openaiPricing.ts');

const catalog = JSON.parse(readFileSync(catalogPath, 'utf8'));

const entries = Object.entries(catalog.models)
  .filter(
    ([, entry]) =>
      entry.pricing?.input_per_million_tokens != null &&
      entry.pricing?.output_per_million_tokens != null,
  )
  .sort(([a], [b]) => (a < b ? -1 : 1));

const lines = entries.map(([name, entry]) => {
  const pricing = entry.pricing;
  const fields = [
    `input: ${pricing.input_per_million_tokens}`,
    `output: ${pricing.output_per_million_tokens}`,
  ];
  if (pricing.cache_read_per_million_tokens != null) {
    fields.push(`cacheRead: ${pricing.cache_read_per_million_tokens}`);
  }
  if (pricing.cache_write_per_million_tokens != null) {
    fields.push(`cacheWrite: ${pricing.cache_write_per_million_tokens}`);
  }
  return `  '${name}': { ${fields.join(', ')} },`;
});

const content = `/**
 * GENERATED FILE - DO NOT EDIT BY HAND.
 *
 * Snapshot of OpenAI per-model pricing from MLflow's bundled model catalog
 * (mlflow/utils/model_catalog/openai.json). Regenerate with:
 *
 *   npm run sync:pricing
 *
 * tests/openaiPricing.test.ts fails when this file drifts from the catalog.
 */

/** Per-million-token USD rates for one OpenAI model. */
export interface OpenAIModelRate {
  /** USD per million regular (non-cached) input tokens. */
  input: number;
  /** USD per million output tokens. */
  output: number;
  /** USD per million cache-read (cached) input tokens. */
  cacheRead?: number;
  /** USD per million cache-write input tokens (unused by OpenAI today). */
  cacheWrite?: number;
}

export const OPENAI_MODEL_RATES: Readonly<Record<string, OpenAIModelRate>> = {
${lines.join('\n')}
};
`;

writeFileSync(outputPath, content);
console.log(`Wrote ${entries.length} models to ${outputPath}`);
