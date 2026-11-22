import type { RunEntity } from '../../../types';

/** Prefer bounded assessment means, preserving table order and excluding missing/non-finite values. */
export const getDefaultChartMetricKeys = (runs: RunEntity[], metricKeys: string[], limit = 5): string[] => {
  const values = new Map<string, { sum: number; count: number }>();
  for (const run of runs) {
    for (const metric of run.data.metrics ?? []) {
      if (!Number.isFinite(metric.value)) continue;
      const aggregate = values.get(metric.key) ?? { sum: 0, count: 0 };
      aggregate.sum += metric.value;
      aggregate.count++;
      values.set(metric.key, aggregate);
    }
  }
  const bounded = metricKeys.filter((key) => {
    const aggregate = values.get(key);
    if (!aggregate) return false;
    const mean = aggregate.sum / aggregate.count;
    return mean >= 0 && mean <= 1;
  });
  return (bounded.length ? bounded : metricKeys.filter((key) => values.has(key))).slice(0, limit);
};
