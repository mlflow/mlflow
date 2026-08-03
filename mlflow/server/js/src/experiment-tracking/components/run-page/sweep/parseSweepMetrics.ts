import type { MetricEntitiesByName } from '../../../types';

/**
 * Parses the summary metrics that `mlflow.genai.evaluate_sweep` flattens onto its parent run
 * into a config x scorer comparison.
 *
 * The backend logs one metric per statistic, keyed as:
 * - `{config}/{scorer}/mean|ci_low|ci_high|std`
 * - `{config}/latency_p50_ms|latency_p90_ms|latency_p95_ms|latency_p99_ms`
 * - `{config}/cost_per_request_usd`
 *
 * Config and scorer names may themselves contain `/` (MLflow allows it in metric keys), so a
 * scorer key can't be split on `/` alone. Latency and cost keys are unambiguous — their suffix is
 * a single segment — so config names are collected from those first and then used to split the
 * scorer keys. See `resolveConfigAndScorer` for the fallback when a config has neither.
 */

export const LATENCY_PERCENTILE_SUFFIXES = {
  '/latency_p50_ms': 'p50',
  '/latency_p90_ms': 'p90',
  '/latency_p95_ms': 'p95',
  '/latency_p99_ms': 'p99',
} as const;

export const COST_PER_REQUEST_SUFFIX = '/cost_per_request_usd';

const SCORER_STAT_SUFFIXES = {
  '/mean': 'mean',
  '/ci_low': 'ciLow',
  '/ci_high': 'ciHigh',
  '/std': 'std',
} as const;

export interface SweepScorerStats {
  /** Mean scorer value across repeats. */
  mean?: number;
  /** Lower bound of the 95% confidence interval. */
  ciLow?: number;
  /** Upper bound of the 95% confidence interval. */
  ciHigh?: number;
  /** Standard deviation of the per-repeat means. */
  std?: number;
}

export interface SweepLatencyStats {
  p50?: number;
  p90?: number;
  p95?: number;
  p99?: number;
}

export interface SweepConfigRow {
  config: string;
  scorersByName: Record<string, SweepScorerStats>;
  latency?: SweepLatencyStats;
  costPerRequestUsd?: number;
}

export interface SweepComparison {
  configs: SweepConfigRow[];
  scorerNames: string[];
}

/**
 * Splits a `{config}/{scorer}` key using the config names discovered from the unambiguous
 * latency/cost keys, preferring the longest match so a config named `a` doesn't claim a key
 * belonging to a config named `a/b`. Falls back to treating the last segment as the scorer, which
 * is correct unless a scorer name contains `/`.
 */
const resolveConfigAndScorer = (key: string, knownConfigs: Set<string>): [string, string] => {
  let matchedConfig: string | undefined;
  for (const config of knownConfigs) {
    if (key.startsWith(`${config}/`) && (matchedConfig === undefined || config.length > matchedConfig.length)) {
      matchedConfig = config;
    }
  }
  if (matchedConfig !== undefined) {
    return [matchedConfig, key.slice(matchedConfig.length + 1)];
  }
  const lastSlash = key.lastIndexOf('/');
  return [key.slice(0, lastSlash), key.slice(lastSlash + 1)];
};

const findSuffix = <T extends string>(key: string, suffixes: Record<string, T>): [string, T] | undefined => {
  for (const [suffix, field] of Object.entries(suffixes) as [string, T][]) {
    // A bare suffix (e.g. the metric key "/mean") leaves no room for a config name.
    if (key.endsWith(suffix) && key.length > suffix.length) {
      return [key.slice(0, -suffix.length), field];
    }
  }
  return undefined;
};

export const parseSweepMetrics = (latestMetrics: MetricEntitiesByName | undefined): SweepComparison => {
  const rowsByConfig = new Map<string, SweepConfigRow>();
  const getRow = (config: string): SweepConfigRow => {
    let row = rowsByConfig.get(config);
    if (!row) {
      row = { config, scorersByName: {} };
      rowsByConfig.set(config, row);
    }
    return row;
  };

  const entries = Object.entries(latestMetrics ?? {}).map(
    ([key, metric]) => [key, Number(metric?.value)] as [string, number],
  );
  // Metrics can arrive as non-numeric (or absent) values; those carry no comparison signal.
  const numericEntries = entries.filter(([, value]) => Number.isFinite(value));

  // Pass 1: latency and cost, which also establish the known config names.
  const scorerEntries: [configAndScorer: string, field: keyof SweepScorerStats, value: number][] = [];
  for (const [key, value] of numericEntries) {
    const latencyMatch = findSuffix(key, LATENCY_PERCENTILE_SUFFIXES);
    if (latencyMatch) {
      const [config, percentile] = latencyMatch;
      const row = getRow(config);
      row.latency = { ...row.latency, [percentile]: value };
      continue;
    }
    if (key.endsWith(COST_PER_REQUEST_SUFFIX) && key.length > COST_PER_REQUEST_SUFFIX.length) {
      getRow(key.slice(0, -COST_PER_REQUEST_SUFFIX.length)).costPerRequestUsd = value;
      continue;
    }
    const scorerMatch = findSuffix(key, SCORER_STAT_SUFFIXES);
    if (scorerMatch) {
      const [configAndScorer, field] = scorerMatch;
      scorerEntries.push([configAndScorer, field, value]);
    }
  }

  // Pass 2: scorer statistics, split against the config names found above.
  const knownConfigs = new Set(rowsByConfig.keys());
  const scorerNames = new Set<string>();
  for (const [configAndScorer, field, value] of scorerEntries) {
    if (configAndScorer.lastIndexOf('/') <= 0) {
      // No `{config}/{scorer}` structure — not a sweep summary metric.
      continue;
    }
    const [config, scorer] = resolveConfigAndScorer(configAndScorer, knownConfigs);
    if (!config || !scorer) {
      continue;
    }
    const row = getRow(config);
    row.scorersByName[scorer] = { ...row.scorersByName[scorer], [field]: value };
    scorerNames.add(scorer);
  }

  // Sorted so the table is stable regardless of the order the API returns metrics in.
  return {
    configs: [...rowsByConfig.values()]
      .filter((row) => Object.keys(row.scorersByName).length > 0 || row.latency || row.costPerRequestUsd !== undefined)
      .sort((a, b) => a.config.localeCompare(b.config)),
    scorerNames: [...scorerNames].sort((a, b) => a.localeCompare(b)),
  };
};

/**
 * Names of the configs with the highest mean for `scorer`.
 *
 * Higher is assumed better, matching `SweepResult.best()`'s default. Returns every config whose
 * confidence interval overlaps the leader's, since those aren't distinguishable at 95% confidence
 * — a single winner is only claimed when the intervals are actually disjoint.
 */
export const findBestConfigs = (configs: SweepConfigRow[], scorer: string): string[] => {
  const scored = configs.flatMap((row) => {
    const { mean, ciLow, ciHigh } = row.scorersByName[scorer] ?? {};
    return mean === undefined ? [] : [{ config: row.config, mean, ciLow, ciHigh }];
  });

  if (scored.length === 0) {
    return [];
  }

  const leader = scored.reduce((best, entry) => (entry.mean > best.mean ? entry : best));

  return scored
    .filter(({ config, ciHigh }) => {
      if (config === leader.config) {
        return true;
      }
      // Without both intervals there's no overlap to test, so only the strict leader is best.
      if (leader.ciLow === undefined || ciHigh === undefined) {
        return false;
      }
      return ciHigh >= leader.ciLow;
    })
    .map(({ config }) => config);
};
