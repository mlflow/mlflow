/**
 * Cost estimation for Claude models, used to populate `mlflow.llm.cost`.
 *
 * The Claude Code transcript carries the model and token usage but no cost, and
 * neither the MLflow TypeScript SDK nor the Databricks tracing backend computes
 * cost from usage (unlike the OSS Python path, which does it server-side or via
 * the client on Databricks). So the plugin computes it here and writes it onto the
 * spans/trace in the same shape MLflow's Python cost pipeline produces.
 *
 * Rates come from MLflow's model catalog, the same data the Python cost
 * pipeline reads: the published catalog fetched at runtime when available
 * (see `modelCatalog.ts`), falling back to the snapshot bundled in
 * `anthropicPricing.ts` at development time (`npm run sync:pricing`).
 */
import type { TokenUsage } from './types.js';
import { ANTHROPIC_MODEL_RATES, type AnthropicModelRate } from './anthropicPricing.js';

export type ModelRates = Readonly<Record<string, AnthropicModelRate>>;

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

let activeRates: ModelRates = ANTHROPIC_MODEL_RATES;

/**
 * Replace the rate table used by calculateCost — processTranscript calls this
 * with rates resolved from the remote model catalog. Passing null restores the
 * bundled snapshot.
 */
export function setModelRates(rates: ModelRates | null): void {
  activeRates = rates ?? ANTHROPIC_MODEL_RATES;
}

function lookupRate(model: string): AnthropicModelRate | undefined {
  const exact = activeRates[model];
  if (exact) {
    return exact;
  }
  // A dated snapshot newer than the catalog (e.g. claude-opus-4-8-20260901)
  // falls back to its undated family alias.
  const undated = model.replace(/-\d{8}$/, '');
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

  const cacheRead = usage.cache_read_input_tokens || 0;
  const cacheWrite = usage.cache_creation_input_tokens || 0;

  // Cache rates come from the catalog per model; fall back to the base input
  // rate when missing, matching Python's cost_per_token.
  const inputCost =
    inputTokens * rate.input * PER_MTOK +
    cacheRead * (rate.cacheRead ?? rate.input) * PER_MTOK +
    cacheWrite * (rate.cacheWrite ?? rate.input) * PER_MTOK;
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
