import { describe, test, expect } from '@jest/globals';

import type { MetricEntitiesByName } from '../../../types';
import { findBestConfigs, parseSweepMetrics } from './parseSweepMetrics';

/**
 * Guards the contract between `mlflow.genai.evaluate_sweep` and this parser.
 *
 * The fixture below is verbatim output from a real sweep — `dev/sweep_contract_check.py` runs
 * `evaluate_sweep` with traced predict functions and a code-based scorer, then prints every metric
 * key logged to the parent run. If the backend renames a metric suffix, this test fails, which is
 * the signal that `parseSweepMetrics` needs updating.
 *
 * Two configs x one scorer x 3 repeats, over 3 rows. `cost` is absent because the predict
 * functions report no token usage, which is also what happens with a provider that doesn't
 * return usage data.
 */
const REAL_SWEEP_METRICS = {
  'fast-model/exact_match/ci_high': 0.8112214085284226,
  'fast-model/exact_match/ci_low': 0.2666513833570395,
  'fast-model/exact_match/mean': 0.5555555555555556,
  'fast-model/exact_match/std': 0.49690399499995325,
  'fast-model/latency_p50_ms': 10.0,
  'fast-model/latency_p90_ms': 12.2,
  'fast-model/latency_p95_ms': 12.6,
  'fast-model/latency_p99_ms': 12.92,
  'slow-model/exact_match/ci_high': 0.9999999999999999,
  'slow-model/exact_match/ci_low': 0.7008551269691575,
  'slow-model/exact_match/mean': 1.0,
  'slow-model/exact_match/std': 0.0,
  'slow-model/latency_p50_ms': 50.0,
  'slow-model/latency_p90_ms': 53.0,
  'slow-model/latency_p95_ms': 53.0,
  'slow-model/latency_p99_ms': 53.0,
} satisfies Record<string, number>;

const asLatestMetrics = (values: Record<string, number>): MetricEntitiesByName =>
  Object.fromEntries(
    Object.entries(values).map(([key, value]) => [key, { key, value, step: 0, timestamp: 0 }]),
  ) as MetricEntitiesByName;

describe('parseSweepMetrics against real evaluate_sweep output', () => {
  const { configs, scorerNames } = parseSweepMetrics(asLatestMetrics(REAL_SWEEP_METRICS));

  test('recovers every config and scorer the sweep logged', () => {
    expect(configs.map(({ config }) => config)).toEqual(['fast-model', 'slow-model']);
    expect(scorerNames).toEqual(['exact_match']);
  });

  test('groups all four scorer statistics under the right config', () => {
    expect(configs[0].scorersByName['exact_match']).toEqual({
      mean: 0.5555555555555556,
      ciLow: 0.2666513833570395,
      ciHigh: 0.8112214085284226,
      std: 0.49690399499995325,
    });
    expect(configs[1].scorersByName['exact_match']).toEqual({
      mean: 1.0,
      ciLow: 0.7008551269691575,
      ciHigh: 0.9999999999999999,
      std: 0.0,
    });
  });

  test('recovers the latency percentiles measured from trace durations', () => {
    expect(configs[0].latency).toEqual({ p50: 10.0, p90: 12.2, p95: 12.6, p99: 12.92 });
    expect(configs[1].latency).toEqual({ p50: 50.0, p90: 53.0, p95: 53.0, p99: 53.0 });
  });

  test('leaves cost undefined when no trace reported token usage', () => {
    expect(configs.every(({ costPerRequestUsd }) => costPerRequestUsd === undefined)).toBe(true);
  });

  test('tags both configs as best because their intervals overlap', () => {
    // slow-model has the higher mean (1.0 vs 0.556), but fast-model's ci_high (0.811) exceeds
    // slow-model's ci_low (0.701), so 3 repeats are not enough to separate them.
    expect(findBestConfigs(configs, 'exact_match').sort()).toEqual(['fast-model', 'slow-model']);
  });

  test('drops only the renamed statistic if the backend renames one suffix', () => {
    const renamed = Object.fromEntries(
      Object.entries(REAL_SWEEP_METRICS).map(([key, value]) => [key.replace(/\/mean$/, '/average'), value]),
    );
    const parsed = parseSweepMetrics(asLatestMetrics(renamed));

    // The scorer is still found via its other statistics, so the table keeps rendering and only
    // the Mean cell falls back to a placeholder. Losing one suffix degrades a column, not the tab.
    expect(parsed.scorerNames).toEqual(['exact_match']);
    expect(parsed.configs[0].scorersByName['exact_match'].mean).toBeUndefined();
    expect(parsed.configs[0].scorersByName['exact_match'].ciLow).toBe(0.2666513833570395);
    // Without a mean there is nothing to rank, so no config is tagged best.
    expect(findBestConfigs(parsed.configs, 'exact_match')).toEqual([]);
  });

  test('drops a scorer entirely only when every statistic suffix changes', () => {
    const renamed = Object.fromEntries(
      Object.entries(REAL_SWEEP_METRICS).map(([key, value]) => [
        key.replace(/\/(mean|ci_low|ci_high|std)$/, '/x_$1'),
        value,
      ]),
    );
    const parsed = parseSweepMetrics(asLatestMetrics(renamed));

    expect(parsed.scorerNames).toEqual([]);
    // Latency keys are untouched, so the configs survive and still show cost/latency.
    expect(parsed.configs.map(({ config }) => config)).toEqual(['fast-model', 'slow-model']);
  });
});
