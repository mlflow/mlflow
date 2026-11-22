import type { RunEntity } from '../../types';

const getEvalRunMetricValue = (run: RunEntity, metricKey: string): number | undefined =>
  run.data.metrics?.find((metric) => metric.key === metricKey)?.value;

// Groups are flat (grouping disables parent/child nesting), so the mean covers every run in the group.
export const getEvalRunsGroupMetricMean = (runs: RunEntity[], metricKey: string): number | undefined => {
  const values = runs
    .map((run) => getEvalRunMetricValue(run, metricKey))
    .filter((value): value is number => Number.isFinite(value));
  return values.length > 0 ? values.reduce((sum, value) => sum + value, 0) / values.length : undefined;
};
