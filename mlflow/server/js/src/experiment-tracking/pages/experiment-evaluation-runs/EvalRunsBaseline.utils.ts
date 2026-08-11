import type { RunEntity } from '../../types';

/**
 * The baseline is stored as a single experiment tag, so it is shared by every
 * viewer of the experiment rather than being per-user. The value packs the
 * setter and timestamp alongside the run id so the UI can explain who chose it.
 */
export const EVAL_RUNS_BASELINE_TAG = 'mlflow.evaluation.baselineRunId';

/** A delta this small is noise, and must not render as a coloured arrow. */
export const EVAL_RUNS_DELTA_NOISE_FLOOR = 0.0005;

export interface EvalRunsBaselineTagValue {
  runUuid: string;
  setBy?: string;
  setAt?: number;
}

export const serializeBaselineTag = (value: EvalRunsBaselineTagValue): string => JSON.stringify(value);

/**
 * Older values (and anything written by the API directly) may be a bare run id,
 * so fall back to treating the whole string as the run uuid.
 */
export const parseBaselineTag = (raw: string | undefined | null): EvalRunsBaselineTagValue | undefined => {
  if (!raw) {
    return undefined;
  }
  try {
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed.runUuid === 'string') {
      return parsed;
    }
  } catch {
    // not JSON — fall through to the bare-run-id form
  }
  return { runUuid: raw };
};

const LOWER_IS_BETTER_PATTERNS = [/latency/i, /duration/i, /cost/i, /error/i, /_ms$/i, /token/i];

/**
 * MLflow has no per-scorer `greater_is_better` metadata, so direction is
 * inferred from the metric name. Wrong for a scorer named unexpectedly, but
 * without it `latency ▲` renders green while the app got slower.
 */
export const isLowerBetterMetric = (metricKey: string): boolean =>
  LOWER_IS_BETTER_PATTERNS.some((pattern) => pattern.test(metricKey));

export type EvalRunsDeltaDirection = 'better' | 'worse' | 'neutral';

export interface EvalRunsDelta {
  value: number;
  direction: EvalRunsDeltaDirection;
}

export const getEvalRunsDelta = (
  metricKey: string,
  value: number | undefined,
  baselineValue: number | undefined,
): EvalRunsDelta | undefined => {
  if (!Number.isFinite(value) || !Number.isFinite(baselineValue)) {
    return undefined;
  }
  const delta = (value as number) - (baselineValue as number);
  if (Math.abs(delta) < EVAL_RUNS_DELTA_NOISE_FLOOR) {
    return { value: delta, direction: 'neutral' };
  }
  const improved = isLowerBetterMetric(metricKey) ? delta < 0 : delta > 0;
  return { value: delta, direction: improved ? 'better' : 'worse' };
};

export const getRunMetricValue = (run: RunEntity | undefined, metricKey: string): number | undefined => {
  const metric = run?.data?.metrics?.find(({ key }) => key === metricKey);
  return metric ? Number(metric.value) : undefined;
};

/** Numeric mean over the runs that actually reported the metric. */
export const getMetricMean = (runs: RunEntity[], metricKey: string): number | undefined => {
  const values = runs.map((run) => getRunMetricValue(run, metricKey)).filter((v): v is number => Number.isFinite(v));
  if (values.length === 0) {
    return undefined;
  }
  return values.reduce((sum, v) => sum + v, 0) / values.length;
};

/**
 * Metrics render to 3 decimals below 10 and 1 above, so `0.912` and `529.4`
 * both stay inside the 96px column.
 */
export const formatEvalRunsMetric = (value: number): string =>
  Math.abs(value) >= 10 ? value.toFixed(1) : value.toFixed(3);
