/**
 * Cost estimation for Claude models, used to populate `mlflow.llm.cost`.
 *
 * The Claude Code transcript carries the model and token usage but no cost, and
 * neither the MLflow TypeScript SDK nor the Databricks tracing backend computes
 * cost from usage (unlike the OSS Python path, which does it server-side or via
 * the client on Databricks). So the plugin computes it here and writes it onto the
 * spans/trace in the same shape MLflow's Python cost pipeline produces.
 *
 * Rates mirror MLflow's bundled provider pricing (`mlflow/utils/providers`).
 * Anthropic prices cached tokens at fixed multiples of the base input rate: cache
 * creation (write) = 1.25x and cache read = 0.1x. These prices are hardcoded and
 * may need updating as Anthropic changes pricing or ships new models.
 */
import type { TokenUsage } from './types.js';

/** Span attribute holding a single call's cost, matching Python's SpanAttributeKey.LLM_COST. */
export const LLM_COST_ATTRIBUTE = 'mlflow.llm.cost';

/** Trace metadata holding the aggregated cost, matching Python's TraceMetadataKey.COST. */
export const TRACE_COST_METADATA = 'mlflow.trace.cost';

export interface LlmCost {
  input_cost: number;
  output_cost: number;
  total_cost: number;
}

interface BaseRate {
  /** USD per input token (per million tokens; scaled by PER_MTOK). */
  input: number;
  /** USD per output token (per million tokens; scaled by PER_MTOK). */
  output: number;
}

const PER_MTOK = 1e-6;

// Anthropic prices cached tokens as multiples of the base input rate.
const CACHE_WRITE_MULTIPLIER = 1.25;
const CACHE_READ_MULTIPLIER = 0.1;

// Prefix -> per-MTok base rates, most specific first (matched with startsWith).
const CLAUDE_MODEL_RATES: ReadonlyArray<readonly [string, BaseRate]> = [
  ['claude-opus-4', { input: 15, output: 75 }],
  ['claude-sonnet-4', { input: 3, output: 15 }],
  ['claude-haiku-4', { input: 1, output: 5 }],
  ['claude-3-7-sonnet', { input: 3, output: 15 }],
  ['claude-3-5-sonnet', { input: 3, output: 15 }],
  ['claude-3-5-haiku', { input: 0.8, output: 4 }],
  ['claude-3-opus', { input: 15, output: 75 }],
  ['claude-3-haiku', { input: 0.25, output: 1.25 }],
];

function lookupRate(model: string): BaseRate | undefined {
  return CLAUDE_MODEL_RATES.find(([prefix]) => model.startsWith(prefix))?.[1];
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

  const inputRate = rate.input * PER_MTOK;
  const inputCost =
    inputTokens * inputRate +
    cacheRead * inputRate * CACHE_READ_MULTIPLIER +
    cacheWrite * inputRate * CACHE_WRITE_MULTIPLIER;
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
