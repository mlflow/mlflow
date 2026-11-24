const EVAL_RUNS_MAX_DECIMAL_PLACES = 4;

const EVAL_RUNS_AXIS_MAX_DECIMAL_PLACES = 2;

const evalRunsNumberFormatter = new Intl.NumberFormat('en-US', {
  maximumFractionDigits: EVAL_RUNS_MAX_DECIMAL_PLACES,
  useGrouping: false,
});

const evalRunsAxisNumberFormatter = new Intl.NumberFormat('en-US', {
  maximumFractionDigits: EVAL_RUNS_AXIS_MAX_DECIMAL_PLACES,
  useGrouping: false,
});

/**
 * Formats an evaluation-run number for display without changing the value used by the table or
 * chart. Trailing zeroes are removed so values that already have fewer decimal places stay compact.
 */
export const formatEvalRunsNumericValue = (value: number): string => {
  if (!Number.isFinite(value)) {
    return String(value);
  }
  return evalRunsNumberFormatter.format(value);
};

/** Chart axis ticks use at most two decimal places so the labels fit the fixed-width axis. */
export const formatEvalRunsAxisValue = (value: number): string => {
  if (!Number.isFinite(value)) {
    return String(value);
  }
  return evalRunsAxisNumberFormatter.format(value);
};
