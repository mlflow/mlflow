import { describe, test, expect } from '@jest/globals';

import type { MetricEntitiesByName } from '../../types';
import { findBestConfigs, parseSweepMetrics } from './parseSweepMetrics';

const metrics = (values: Record<string, number>): MetricEntitiesByName =>
  Object.fromEntries(
    Object.entries(values).map(([key, value]) => [key, { key, value, step: 0, timestamp: 0 }]),
  ) as MetricEntitiesByName;

describe('parseSweepMetrics', () => {
  test('returns nothing for missing or unrelated metrics', () => {
    expect(parseSweepMetrics(undefined)).toEqual({ configs: [], scorerNames: [] });
    expect(parseSweepMetrics(metrics({}))).toEqual({ configs: [], scorerNames: [] });
    expect(parseSweepMetrics(metrics({ accuracy: 0.9, loss: 0.1 }))).toEqual({ configs: [], scorerNames: [] });
  });

  test('groups scorer statistics, latency, and cost by config', () => {
    const { configs, scorerNames } = parseSweepMetrics(
      metrics({
        'gpt-4o/correctness/mean': 0.9,
        'gpt-4o/correctness/ci_low': 0.85,
        'gpt-4o/correctness/ci_high': 0.95,
        'gpt-4o/correctness/std': 0.02,
        'gpt-4o/latency_p50_ms': 120,
        'gpt-4o/latency_p90_ms': 200,
        'gpt-4o/latency_p95_ms': 240,
        'gpt-4o/latency_p99_ms': 300,
        'gpt-4o/cost_per_request_usd': 0.0123,
        'claude/correctness/mean': 0.8,
        'claude/latency_p50_ms': 90,
      }),
    );

    expect(scorerNames).toEqual(['correctness']);
    // Sorted by config name, so the table order does not depend on the API response order.
    expect(configs.map(({ config }) => config)).toEqual(['claude', 'gpt-4o']);
    expect(configs[1]).toEqual({
      config: 'gpt-4o',
      scorersByName: { correctness: { mean: 0.9, ciLow: 0.85, ciHigh: 0.95, std: 0.02 } },
      latency: { p50: 120, p90: 200, p95: 240, p99: 300 },
      costPerRequestUsd: 0.0123,
    });
    expect(configs[0]).toEqual({
      config: 'claude',
      scorersByName: { correctness: { mean: 0.8 } },
      latency: { p50: 90 },
    });
  });

  test('collects every scorer across configs', () => {
    const { configs, scorerNames } = parseSweepMetrics(
      metrics({
        'a/safety/mean': 1,
        'a/correctness/mean': 0.5,
        'b/relevance/mean': 0.7,
      }),
    );

    expect(scorerNames).toEqual(['correctness', 'relevance', 'safety']);
    expect(Object.keys(configs[0].scorersByName).sort()).toEqual(['correctness', 'safety']);
  });

  test('splits config names containing a slash using the latency keys', () => {
    const { configs, scorerNames } = parseSweepMetrics(
      metrics({
        'team/gpt-4o/latency_p50_ms': 100,
        'team/gpt-4o/correctness/mean': 0.9,
      }),
    );

    expect(scorerNames).toEqual(['correctness']);
    expect(configs).toHaveLength(1);
    expect(configs[0].config).toBe('team/gpt-4o');
    expect(configs[0].scorersByName).toEqual({ correctness: { mean: 0.9 } });
  });

  test('prefers the longest matching config name', () => {
    const { configs } = parseSweepMetrics(
      metrics({
        'a/latency_p50_ms': 100,
        'a/b/latency_p50_ms': 200,
        'a/b/correctness/mean': 0.9,
      }),
    );

    const nested = configs.find(({ config }) => config === 'a/b');
    expect(nested?.scorersByName).toEqual({ correctness: { mean: 0.9 } });
    // The key belongs to 'a/b', so the shorter 'a' must not also claim a scorer from it.
    expect(configs.find(({ config }) => config === 'a')?.scorersByName).toEqual({});
  });

  test('falls back to the last segment as the scorer when a config has no latency or cost', () => {
    const { configs, scorerNames } = parseSweepMetrics(metrics({ 'gpt-4o/correctness/mean': 0.9 }));

    expect(scorerNames).toEqual(['correctness']);
    expect(configs[0].config).toBe('gpt-4o');
  });

  test('ignores keys with no config/scorer structure and non-numeric values', () => {
    const { configs } = parseSweepMetrics(
      metrics({
        '/mean': 0.5,
        mean: 0.5,
        'orphan/mean': 0.5,
      }),
    );
    expect(configs).toEqual([]);

    const nonNumeric = parseSweepMetrics({
      'gpt-4o/correctness/mean': { key: 'gpt-4o/correctness/mean', value: NaN, step: 0, timestamp: 0 },
    } as MetricEntitiesByName);
    expect(nonNumeric.configs).toEqual([]);
  });

  test('keeps a config that only reported latency or cost', () => {
    const { configs, scorerNames } = parseSweepMetrics(
      metrics({ 'gpt-4o/latency_p50_ms': 100, 'gpt-4o/cost_per_request_usd': 0.01 }),
    );

    expect(scorerNames).toEqual([]);
    expect(configs).toHaveLength(1);
    expect(configs[0]).toEqual({
      config: 'gpt-4o',
      scorersByName: {},
      latency: { p50: 100 },
      costPerRequestUsd: 0.01,
    });
  });
});

describe('findBestConfigs', () => {
  const configOf = (config: string, mean: number, ciLow?: number, ciHigh?: number) => ({
    config,
    scorersByName: { correctness: { mean, ciLow, ciHigh } },
  });

  test('returns the single highest mean when the intervals are disjoint', () => {
    const configs = [configOf('a', 0.9, 0.88, 0.92), configOf('b', 0.5, 0.48, 0.52)];
    expect(findBestConfigs(configs, 'correctness')).toEqual(['a']);
  });

  test('returns every config whose interval overlaps the leader', () => {
    const configs = [configOf('a', 0.9, 0.7, 0.99), configOf('b', 0.85, 0.65, 0.95), configOf('c', 0.2, 0.1, 0.3)];
    expect(findBestConfigs(configs, 'correctness').sort()).toEqual(['a', 'b']);
  });

  test('returns only the leader when the intervals are unavailable', () => {
    const configs = [configOf('a', 0.9), configOf('b', 0.85)];
    expect(findBestConfigs(configs, 'correctness')).toEqual(['a']);
  });

  test('ignores configs missing the scorer and returns nothing when none have it', () => {
    const configs = [configOf('a', 0.9, 0.8, 1), { config: 'b', scorersByName: {} }];
    expect(findBestConfigs(configs, 'correctness')).toEqual(['a']);
    expect(findBestConfigs(configs, 'safety')).toEqual([]);
    expect(findBestConfigs([], 'correctness')).toEqual([]);
  });
});
