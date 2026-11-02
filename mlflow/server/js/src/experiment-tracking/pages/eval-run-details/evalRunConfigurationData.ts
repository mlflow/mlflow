import { getTraceCost, getTraceTokenUsage, type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import type { UseGetRunQueryDataResponse } from '../../components/run-page/hooks/useGetRunQuery';

export interface EvalRunConfigurationMetadata {
  runName?: string;
  dataset?: string;
  createdAt?: number;
}

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

export type EvalRunConfigurationMetric = 'traceCount' | 'totalTokens' | 'totalCost' | 'errors';

export const getEvalRunConfigurationMetadata = (run: UseGetRunQueryDataResponse): EvalRunConfigurationMetadata => {
  const startTime = run?.info?.startTime;
  const createdAt = startTime ? Number(startTime) : undefined;
  return {
    runName: run?.info?.runName ?? undefined,
    dataset: run?.inputs?.datasetInputs?.[0]?.dataset?.name ?? undefined,
    createdAt: createdAt !== undefined && Number.isFinite(createdAt) ? createdAt : undefined,
  };
};

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

export const isEvalRunConfigurationMetricComplete = (
  stats: EvalRunConfigurationTraceStats | undefined,
  metric: EvalRunConfigurationMetric,
): boolean =>
  Boolean(
    stats &&
    !stats.isTruncated &&
    stats[metric] !== undefined &&
    (metric !== 'totalTokens' || stats.tracesWithTokens === stats.traceCount) &&
    (metric !== 'totalCost' || stats.tracesWithCost === stats.traceCount),
  );

export const getEvalRunConfigurationMinimum = (
  stats: (EvalRunConfigurationTraceStats | undefined)[],
  metric: EvalRunConfigurationMetric,
): number | undefined => {
  if (
    metric === 'traceCount' ||
    stats.length < 2 ||
    !stats.every((run) => isEvalRunConfigurationMetricComplete(run, metric))
  ) {
    return undefined;
  }
  return Math.min(...stats.map((run) => run?.[metric] ?? Infinity));
};
