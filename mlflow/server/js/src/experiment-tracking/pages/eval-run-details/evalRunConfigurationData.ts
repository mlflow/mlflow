import { getTraceCost, getTraceTokenUsage, type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';

export interface EvalRunConfigurationTraceStats {
  traceCount: number;
  totalTokens?: number;
  totalCost?: number;
  errors: number;
  tracesWithTokens: number;
  tracesWithCost: number;
  isTruncated: boolean;
}

export interface EvalRunSidebarTraceStats extends EvalRunConfigurationTraceStats {
  hasPartialUsage: boolean;
}

const isValidUsage = (value: number | undefined): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0;

export const aggregateEvalRunConfigurationTraces = (
  traces: ModelTraceInfoV3[],
  isTruncated: boolean,
): EvalRunConfigurationTraceStats => {
  let totalTokens = 0;
  let totalCost = 0;
  let tracesWithTokens = 0;
  let tracesWithCost = 0;
  let errors = 0;
  for (const trace of traces) {
    const tokens = getTraceTokenUsage(trace)?.total_tokens;
    const cost = getTraceCost(trace)?.total_cost;
    if (isValidUsage(tokens)) {
      totalTokens += tokens;
      tracesWithTokens++;
    }
    if (isValidUsage(cost)) {
      totalCost += cost;
      tracesWithCost++;
    }
    if (trace.state === 'ERROR') errors++;
  }
  return {
    traceCount: traces.length,
    totalTokens: tracesWithTokens > 0 && Number.isFinite(totalTokens) ? totalTokens : undefined,
    totalCost: tracesWithCost > 0 && Number.isFinite(totalCost) ? totalCost : undefined,
    errors,
    tracesWithTokens,
    tracesWithCost,
    isTruncated,
  };
};

export const aggregateEvalRunSidebarTraceStats = (
  traces: ModelTraceInfoV3[],
  isTruncated: boolean,
): EvalRunSidebarTraceStats => {
  const traceStats = aggregateEvalRunConfigurationTraces(traces, isTruncated);
  return {
    ...traceStats,
    hasPartialUsage:
      traceStats.traceCount > 0 &&
      (traceStats.tracesWithTokens < traceStats.traceCount || traceStats.tracesWithCost < traceStats.traceCount),
  };
};
