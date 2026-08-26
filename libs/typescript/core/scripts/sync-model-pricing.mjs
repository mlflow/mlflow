import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const packageDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repositoryRoot = resolve(packageDir, '../../..');
const catalogDir = join(repositoryRoot, 'mlflow/utils/model_catalog');
const outputPath = join(packageDir, 'src/core/utils/model_pricing_catalog.ts');

const catalog = {};
for (const filename of readdirSync(catalogDir)
  .filter((name) => name.endsWith('.json'))
  .sort()) {
  const provider = filename.slice(0, -'.json'.length);
  const source = JSON.parse(readFileSync(join(catalogDir, filename), 'utf8'));
  const models = {};

  for (const [model, entry] of Object.entries(source.models ?? {})) {
    const pricing = entry.pricing;
    if (!pricing) {
      continue;
    }

    const rate = {};
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
    catalog[provider] = models;
  }
}

const source = `/**
 * GENERATED FILE - DO NOT EDIT BY HAND.
 *
 * Compact pricing snapshot generated from mlflow/utils/model_catalog/*.json.
 * Regenerate with: npm run sync:model-pricing -C libs/typescript/core
 */

export interface ModelPricing {
  inputPerMillion?: number;
  outputPerMillion?: number;
  cacheReadPerMillion?: number;
  cacheWritePerMillion?: number;
}

export type ModelPricingCatalog = Readonly<
  Record<string, Readonly<Record<string, Readonly<ModelPricing>>>>
>;

// prettier-ignore
export const MODEL_PRICING_CATALOG: ModelPricingCatalog = ${JSON.stringify(catalog)};
`;

writeFileSync(outputPath, source);
