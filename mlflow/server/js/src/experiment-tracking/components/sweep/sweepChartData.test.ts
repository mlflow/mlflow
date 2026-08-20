import { describe, test, expect } from '@jest/globals';

import type { RunEntity } from '../../types';
import { MLFLOW_RUN_TYPE_TAG, MLFLOW_RUN_TYPE_VALUE_GENAI_EVALUATE_SWEEP } from '../../constants';
import { buildSweepChartData, buildTradeoffPoints, isSweepRun, type SweepChartPoint } from './sweepChartData';

const run = (runName: string, runType: string | undefined, metrics: Record<string, number>): RunEntity =>
  ({
    info: { runUuid: `${runName}-uuid`, runName },
    data: {
      params: [],
      tags: runType ? [{ key: MLFLOW_RUN_TYPE_TAG, value: runType }] : [],
      metrics: Object.entries(metrics).map(([key, value]) => ({ key, value, step: 0, timestamp: 0 })),
    },
  }) as unknown as RunEntity;

const sweepRun = (runName: string, metrics: Record<string, number>) =>
  run(runName, MLFLOW_RUN_TYPE_VALUE_GENAI_EVALUATE_SWEEP, metrics);

const TWO_CONFIG_METRICS = {
  'gpt-4o/correctness/mean': 0.91,
  'gpt-4o/correctness/ci_low': 0.88,
  'gpt-4o/correctness/ci_high': 0.94,
  'gpt-4o/latency_p50_ms': 412,
  'gpt-4o/cost_per_request_usd': 0.0128,
  'llama/correctness/mean': 0.62,
  'llama/correctness/ci_low': 0.57,
  'llama/correctness/ci_high': 0.67,
  'llama/latency_p50_ms': 128,
  'llama/cost_per_request_usd': 0.0004,
};

describe('isSweepRun', () => {
  test('matches only runs carrying the sweep run-type tag', () => {
    expect(isSweepRun(sweepRun('a', {}))).toBe(true);
    expect(isSweepRun(run('b', 'genai_evaluate', {}))).toBe(false);
    expect(isSweepRun(run('c', undefined, {}))).toBe(false);
  });
});

describe('buildSweepChartData', () => {
  test('returns nothing when no run is a sweep', () => {
    expect(buildSweepChartData([run('plain', undefined, { accuracy: 0.9 })])).toEqual({
      points: [],
      runSeries: [],
      configNames: [],
      scorerNames: [],
    });
  });

  test('builds one point per configuration of a sweep run', () => {
    const { points, scorerNames } = buildSweepChartData([sweepRun('sweep-1', TWO_CONFIG_METRICS)]);

    expect(scorerNames).toEqual(['correctness']);
    expect(points).toHaveLength(2);
    // A single sweep needs no run prefix, so the label is just the config name.
    expect(points.map((point) => point.label)).toEqual(['gpt-4o', 'llama']);
    expect(points[0]).toMatchObject({
      config: 'gpt-4o',
      runUuid: 'sweep-1-uuid',
      latency: { p50: 412 },
      costPerRequestUsd: 0.0128,
    });
    expect(points[0].scorers['correctness']).toEqual({ mean: 0.91, ciLow: 0.88, ciHigh: 0.94 });
  });

  test('prefixes labels with the run name when several sweeps are plotted', () => {
    const { points } = buildSweepChartData([
      sweepRun('sweep-1', { 'gpt-4o/correctness/mean': 0.9 }),
      sweepRun('sweep-2', { 'gpt-4o/correctness/mean': 0.8 }),
    ]);

    // Same config in two sweeps must stay distinguishable.
    expect(points.map((point) => point.label)).toEqual(['sweep-1 / gpt-4o', 'sweep-2 / gpt-4o']);
  });

  test('ignores non-sweep runs mixed into the selection', () => {
    const { points } = buildSweepChartData([
      run('plain', undefined, { accuracy: 0.5 }),
      sweepRun('sweep-1', { 'gpt-4o/correctness/mean': 0.9 }),
    ]);

    expect(points).toHaveLength(1);
    expect(points[0].label).toBe('gpt-4o');
  });

  test('groups points into one series per sweep run', () => {
    const { runSeries } = buildSweepChartData([
      sweepRun('sweep-1', TWO_CONFIG_METRICS),
      sweepRun('sweep-2', { 'gpt-4o/correctness/mean': 0.8 }),
    ]);

    // One series per run is what lets each run be coloured and legended separately.
    expect(runSeries.map((series) => series.runName)).toEqual(['sweep-1', 'sweep-2']);
    expect(runSeries[0].points).toHaveLength(2);
    expect(runSeries[1].points).toHaveLength(1);
  });

  test('collects config names as shared x-axis categories', () => {
    const { configNames } = buildSweepChartData([
      sweepRun('sweep-1', { 'gpt-4o/correctness/mean': 0.9, 'llama/correctness/mean': 0.6 }),
      // 'claude' appears only in the second run; every config still gets an axis slot.
      sweepRun('sweep-2', { 'gpt-4o/correctness/mean': 0.8, 'claude/correctness/mean': 0.85 }),
    ]);

    expect(configNames).toEqual(['claude', 'gpt-4o', 'llama']);
  });

  test('omits a sweep run whose metrics parsed to nothing', () => {
    const { runSeries } = buildSweepChartData([
      sweepRun('empty-sweep', { unrelated: 1 }),
      sweepRun('sweep-1', { 'gpt-4o/correctness/mean': 0.9 }),
    ]);

    // An empty series would claim a legend entry and a colour for no plotted points.
    expect(runSeries.map((series) => series.runName)).toEqual(['sweep-1']);
  });

  test('collects scorers across runs that swept different scorers', () => {
    const { scorerNames } = buildSweepChartData([
      sweepRun('sweep-1', { 'a/safety/mean': 1 }),
      sweepRun('sweep-2', { 'b/correctness/mean': 0.5 }),
    ]);

    expect(scorerNames).toEqual(['correctness', 'safety']);
  });
});

describe('buildTradeoffPoints', () => {
  const point = (label: string, mean: number, cost: number): SweepChartPoint => ({
    config: label,
    label,
    runUuid: `${label}-uuid`,
    runName: 'sweep-1',
    scorers: { correctness: { mean } },
    costPerRequestUsd: cost,
  });

  const costOf = (p: SweepChartPoint) => p.costPerRequestUsd;

  test('marks both ends of a genuine tradeoff as being on the frontier', () => {
    // Cheap-and-worse and expensive-and-better are both defensible choices.
    const result = buildTradeoffPoints([point('cheap', 0.6, 0.001), point('pricey', 0.9, 0.02)], 'correctness', costOf);

    expect(
      result
        .filter((r) => r.isOnFrontier)
        .map((r) => r.label)
        .sort(),
    ).toEqual(['cheap', 'pricey']);
  });

  test('excludes a configuration that is both worse and more expensive', () => {
    const result = buildTradeoffPoints(
      [point('good-cheap', 0.9, 0.001), point('bad-pricey', 0.5, 0.02)],
      'correctness',
      costOf,
    );

    expect(result.find((r) => r.label === 'good-cheap')?.isOnFrontier).toBe(true);
    expect(result.find((r) => r.label === 'bad-pricey')?.isOnFrontier).toBe(false);
  });

  test('keeps configurations that tie on both axes', () => {
    // Neither dominates the other, so removing either would be arbitrary.
    const result = buildTradeoffPoints([point('a', 0.8, 0.01), point('b', 0.8, 0.01)], 'correctness', costOf);

    expect(result.every((r) => r.isOnFrontier)).toBe(true);
  });

  test('drops configurations missing the scorer or the cost axis', () => {
    const noCost: SweepChartPoint = {
      config: 'no-cost',
      label: 'no-cost',
      runUuid: 'x',
      runName: 'sweep-1',
      scorers: { correctness: { mean: 0.99 } },
    };
    const noScorer: SweepChartPoint = {
      config: 'no-scorer',
      label: 'no-scorer',
      runUuid: 'y',
      runName: 'sweep-1',
      scorers: {},
      costPerRequestUsd: 0.001,
    };

    const result = buildTradeoffPoints([point('ok', 0.8, 0.01), noCost, noScorer], 'correctness', costOf);
    expect(result.map((r) => r.label)).toEqual(['ok']);
  });

  test('supports latency as the tradeoff axis', () => {
    const fast: SweepChartPoint = {
      config: 'fast',
      label: 'fast',
      runUuid: 'a',
      runName: 'sweep-1',
      scorers: { correctness: { mean: 0.7 } },
      latency: { p50: 100 },
    };
    const slow: SweepChartPoint = {
      config: 'slow',
      label: 'slow',
      runUuid: 'b',
      runName: 'sweep-1',
      scorers: { correctness: { mean: 0.6 } },
      latency: { p50: 500 },
    };

    const result = buildTradeoffPoints([fast, slow], 'correctness', (p) => p.latency?.p50);
    // 'slow' is both slower and worse, so it is dominated.
    expect(result.find((r) => r.label === 'fast')?.isOnFrontier).toBe(true);
    expect(result.find((r) => r.label === 'slow')?.isOnFrontier).toBe(false);
  });
});
