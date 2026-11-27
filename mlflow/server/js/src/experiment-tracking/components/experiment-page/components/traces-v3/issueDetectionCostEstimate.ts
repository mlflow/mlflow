/**
 * Client-side heuristic for the pre-run issue detection cost estimate.
 *
 * Baseline of ~$0.005/trace comes from the published cost benchmark (~$0.5 per
 * ~100 traces with the default model); the 0.5x-2x band absorbs variation in
 * model choice and trace size.
 */
const BENCHMARK_COST_PER_TRACE_USD = 0.005;
const ESTIMATE_LOW_MULTIPLIER = 0.5;
const ESTIMATE_HIGH_MULTIPLIER = 2;

export const estimateIssueDetectionCostUsd = (traceCount: number): { low: number; high: number } => ({
  low: traceCount * BENCHMARK_COST_PER_TRACE_USD * ESTIMATE_LOW_MULTIPLIER,
  high: traceCount * BENCHMARK_COST_PER_TRACE_USD * ESTIMATE_HIGH_MULTIPLIER,
});

export const formatEstimatedCostUsd = (cost: number): string =>
  new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2,
  }).format(Math.max(cost, 0.01));
