/**
 * Cost estimation for OpenAI models, used to populate `mlflow.llm.cost`.
 *
 * The Codex transcript carries the model and token usage but no cost, and
 * neither the MLflow TypeScript SDK nor the Databricks tracing backend computes
 * cost from usage (unlike the OSS Python path, which does it server-side or via
 * the client on Databricks). So the plugin computes it here and writes it onto
 * the spans/trace in the same shape MLflow's Python cost pipeline produces.
 *
 * Rates come from MLflow's model catalog, the same data the Python cost
 * pipeline reads: the published catalog fetched at runtime when available
 * (see `modelCatalog.ts`), falling back to the snapshot bundled in
 * `openaiPricing.ts` at development time (`npm run sync:pricing`).
 *
 * OpenAI reports `input_tokens` as the total prompt size and
 * `cached_input_tokens` as the cached subset of it, so cached tokens are priced
 * at the cache-read rate and the remainder at the base input rate. This differs
 * from Anthropic, which reports cache reads separately from `input_tokens`.
 */
import type { TokenUsage } from './types.js';
import { OPENAI_MODEL_RATES, type OpenAIModelRate } from './openaiPricing.js';

export type ModelRates = Readonly<Record<string, OpenAIModelRate>>;

/** Span attribute holding a single call's cost, matching Python's SpanAttributeKey.LLM_COST. */
export const LLM_COST_ATTRIBUTE = 'mlflow.llm.cost';

/** Trace metadata holding the aggregated cost, matching Python's TraceMetadataKey.COST. */
export const TRACE_COST_METADATA = 'mlflow.trace.cost';

export interface LlmCost {
  input_cost: number;
  output_cost: number;
  total_cost: number;
}

const PER_MTOK = 1e-6;

let activeRates: ModelRates = OPENAI_MODEL_RATES;

/**
 * Replace the rate table used by calculateCost — processNotify calls this with
 * rates resolved from the remote model catalog. Passing null restores the
 * bundled snapshot.
 */
export function setModelRates(rates: ModelRates | null): void {
  activeRates = rates ?? OPENAI_MODEL_RATES;
}

function lookupRate(model: string): OpenAIModelRate | undefined {
  const exact = activeRates[model];
  if (exact) {
    return exact;
  }
  // A dated snapshot newer than the catalog (e.g. gpt-5-2026-01-01) falls back
  // to its undated family alias.
  const undated = model.replace(/-\d{4}-\d{2}-\d{2}$/, '');
  return undated === model ? undefined : activeRates[undated];
}

/**
 * Estimate the cost of a single LLM call from its model and token usage.
 * Returns null when the model is unknown or there is no billable base usage.
 */
export function calculateCost(
  model: string | undefined,
  usage: TokenUsage | undefined,
): LlmCost | null {
  if (!model || !usage) {
    return null;
  }
  const rate = lookupRate(model);
  if (!rate) {
    return null;
  }

  const inputTokens = usage.input_tokens || 0;
  const outputTokens = usage.output_tokens || 0;
  // Matches the Python pipeline: no billable base usage -> no cost.
  if (inputTokens === 0 && outputTokens === 0) {
    return null;
  }

  // OpenAI's cached_input_tokens is a subset of input_tokens: price the cached
  // portion at the cache-read rate and the remainder at the base input rate.
  const cached = Math.min(usage.cached_input_tokens || 0, inputTokens);
  const uncachedInput = inputTokens - cached;

  const inputCost =
    uncachedInput * rate.input * PER_MTOK + cached * (rate.cacheRead ?? rate.input) * PER_MTOK;
  const outputCost = outputTokens * rate.output * PER_MTOK;

  return {
    input_cost: inputCost,
    output_cost: outputCost,
    total_cost: inputCost + outputCost,
  };
}

/** Sum per-call costs into a single trace-level total; null when there are none. */
export function sumCosts(costs: readonly LlmCost[]): LlmCost | null {
  if (!costs.length) {
    return null;
  }
  return costs.reduce(
    (acc, c) => ({
      input_cost: acc.input_cost + c.input_cost,
      output_cost: acc.output_cost + c.output_cost,
      total_cost: acc.total_cost + c.total_cost,
    }),
    { input_cost: 0, output_cost: 0, total_cost: 0 },
  );
}
