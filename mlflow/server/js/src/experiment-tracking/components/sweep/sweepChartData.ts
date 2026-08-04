import type { RunEntity } from '../../types';
import { MLFLOW_RUN_TYPE_TAG, MLFLOW_RUN_TYPE_VALUE_GENAI_EVALUATE_SWEEP } from '../../constants';
import { parseSweepMetrics, type SweepConfigRow } from './parseSweepMetrics';

/**
 * Derives the series that the sweep charts plot from `evaluate_sweep` runs.
 *
 * A sweep's configurations all live inside a single run's metrics, so — unlike the generic runs
 * charts, which plot one point per run — these series are per (run, config) pair. When several
 * sweep runs are selected, configs are labelled with their run name so repeat sweeps of the same
 * config stay distinguishable.
 */

/** One swept configuration, ready to plot. */
export interface SweepChartPoint {
  /** Config name as the sweep logged it. This is the x-axis category. */
  config: string;
  /** Display label: the config name, prefixed with the run name when several runs are plotted. */
  label: string;
  runUuid: string;
  runName: string;
  scorers: SweepConfigRow['scorersByName'];
  latency?: SweepConfigRow['latency'];
  costPerRequestUsd?: number;
}

/** The configs of one sweep run, grouped so each run can be plotted as its own series. */
export interface SweepRunSeries {
  runUuid: string;
  runName: string;
  points: SweepChartPoint[];
}

export interface SweepChartData {
  points: SweepChartPoint[];
  /** One entry per sweep run, in the order the runs were given. */
  runSeries: SweepRunSeries[];
  /** Every config name seen across all plotted runs, sorted — the shared x-axis categories. */
  configNames: string[];
  /** Every scorer seen across all plotted runs, sorted. */
  scorerNames: string[];
}

export const isSweepRun = (run: RunEntity): boolean =>
  run.data?.tags?.some(
    (tag) => tag.key === MLFLOW_RUN_TYPE_TAG && tag.value === MLFLOW_RUN_TYPE_VALUE_GENAI_EVALUATE_SWEEP,
  ) ?? false;

/**
 * `parseSweepMetrics` takes the `latestMetrics` shape used by the run page, but the evaluation runs
 * page holds metrics as a list, so convert before reusing the parser.
 */
const metricsByKey = (run: RunEntity) =>
  Object.fromEntries(
    (run.data?.metrics ?? []).map((metric) => [
      metric.key,
      { key: metric.key, value: Number(metric.value), step: 0, timestamp: 0 },
    ]),
  );

export const buildSweepChartData = (runs: RunEntity[]): SweepChartData => {
  const sweepRuns = runs.filter((run) => run.info && isSweepRun(run));
  const prefixWithRunName = sweepRuns.length > 1;

  const points: SweepChartPoint[] = [];
  const runSeries: SweepRunSeries[] = [];
  const configNames = new Set<string>();
  const scorerNames = new Set<string>();

  for (const run of sweepRuns) {
    const parsed = parseSweepMetrics(metricsByKey(run));
    parsed.scorerNames.forEach((scorer) => scorerNames.add(scorer));

    const runName = run.info.runName ?? run.info.runUuid;
    const seriesPoints = parsed.configs.map((config) => {
      configNames.add(config.config);
      return {
        config: config.config,
        label: prefixWithRunName ? `${runName} / ${config.config}` : config.config,
        runUuid: run.info.runUuid,
        runName,
        scorers: config.scorersByName,
        latency: config.latency,
        costPerRequestUsd: config.costPerRequestUsd,
      };
    });

    if (seriesPoints.length > 0) {
      runSeries.push({ runUuid: run.info.runUuid, runName, points: seriesPoints });
      points.push(...seriesPoints);
    }
  }

  return {
    points,
    runSeries,
    configNames: [...configNames].sort((a, b) => a.localeCompare(b)),
    scorerNames: [...scorerNames].sort((a, b) => a.localeCompare(b)),
  };
};

/** A config's score against a cost or latency axis, for the tradeoff charts. */
export interface TradeoffPoint {
  label: string;
  runUuid: string;
  score: number;
  ciLow?: number;
  ciHigh?: number;
  /** Cost per request in USD, or latency in ms, depending on the chart. */
  cost: number;
  /** True when no other config is both cheaper/faster and at least as good. */
  isOnFrontier: boolean;
}

/**
 * Score-vs-cost (or score-vs-latency) points for one scorer, with the Pareto frontier marked.
 *
 * A config is on the frontier when nothing else beats it on both axes at once: anything below the
 * frontier is strictly worse than some alternative, so the frontier is the set of defensible
 * choices. Ties on both axes keep every tied config, since none dominates the other.
 */
export const buildTradeoffPoints = (
  points: SweepChartPoint[],
  scorer: string,
  costOf: (point: SweepChartPoint) => number | undefined,
): TradeoffPoint[] => {
  const candidates = points.flatMap((point) => {
    const stats = point.scorers[scorer];
    const cost = costOf(point);
    if (stats?.mean === undefined || cost === undefined) {
      return [];
    }
    return [
      { label: point.label, runUuid: point.runUuid, score: stats.mean, ciLow: stats.ciLow, ciHigh: stats.ciHigh, cost },
    ];
  });

  return candidates.map((candidate) => ({
    ...candidate,
    isOnFrontier: !candidates.some(
      (other) =>
        other !== candidate &&
        other.cost <= candidate.cost &&
        other.score >= candidate.score &&
        (other.cost < candidate.cost || other.score > candidate.score),
    ),
  }));
};
