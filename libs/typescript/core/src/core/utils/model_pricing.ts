import { MODEL_PRICING_CATALOG, type ModelPricing } from './model_pricing_catalog';

const PER_MILLION = 1 / 1_000_000;

export interface ModelTokenUsage {
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  cache_read_input_tokens?: number;
  cache_creation_input_tokens?: number;
}

export interface ModelCost {
  input_cost: number;
  output_cost: number;
  total_cost: number;
}

/**
 * Estimate model cost using the pricing snapshot bundled with @mlflow/core.
 *
 * This mirrors Python's bundled `mlflow.utils.providers.cost_per_token` path:
 * cached input tokens are removed from regular input usage and priced using
 * their dedicated rates, falling back to the regular input rate when needed.
 */
export function calculateCostByModelAndTokenUsage(
  modelName: string | null | undefined,
  usage: ModelTokenUsage | null | undefined,
  modelProvider?: string | null,
): ModelCost | undefined {
  if (!modelName || !usage) {
    return undefined;
  }

  const inputTokens = validTokenCount(usage.input_tokens);
  const outputTokens = validTokenCount(usage.output_tokens);
  if (inputTokens === 0 && outputTokens === 0) {
    return undefined;
  }

  const pricing = lookupModelPricing(modelName, modelProvider);
  if (!pricing) {
    return undefined;
  }

  const cacheReadTokens = validTokenCount(usage.cache_read_input_tokens);
  const cacheCreationTokens = validTokenCount(usage.cache_creation_input_tokens);
  const regularInputTokens = Math.max(inputTokens - cacheReadTokens - cacheCreationTokens, 0);
  const inputRate = pricing.inputPerMillion ?? 0;
  const outputRate = pricing.outputPerMillion ?? 0;
  const inputCost =
    (regularInputTokens * inputRate +
      cacheReadTokens * (pricing.cacheReadPerMillion ?? inputRate) +
      cacheCreationTokens * (pricing.cacheWritePerMillion ?? inputRate)) *
    PER_MILLION;
  const outputCost = outputTokens * outputRate * PER_MILLION;

  return {
    input_cost: inputCost,
    output_cost: outputCost,
    total_cost: inputCost + outputCost,
  };
}

function lookupModelPricing(
  modelName: string,
  modelProvider?: string | null,
): Readonly<ModelPricing> | undefined {
  const bareModel = modelName.includes('/')
    ? modelName.slice(modelName.indexOf('/') + 1)
    : modelName;

  for (const provider of providerCandidates(modelProvider)) {
    const pricing = MODEL_PRICING_CATALOG[provider]?.[bareModel];
    if (pricing) {
      return pricing;
    }
  }

  for (const models of Object.values(MODEL_PRICING_CATALOG)) {
    const pricing = models[bareModel];
    if (pricing) {
      return pricing;
    }
  }
  return undefined;
}

function providerCandidates(modelProvider?: string | null): string[] {
  if (!modelProvider) {
    return [];
  }
  const provider = modelProvider.trim().toLowerCase();
  if (!provider) {
    return [];
  }

  const family = provider.split(/[./]/, 1)[0];
  return family && family !== provider ? [provider, family] : [provider];
}

function validTokenCount(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : 0;
}
