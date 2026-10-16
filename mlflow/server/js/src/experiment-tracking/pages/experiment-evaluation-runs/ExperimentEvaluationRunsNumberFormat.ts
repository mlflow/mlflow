const EVAL_RUNS_MAX_DECIMAL_PLACES = 4;

const evalRunsNumberFormatter = new Intl.NumberFormat('en-US', {
  maximumFractionDigits: EVAL_RUNS_MAX_DECIMAL_PLACES,
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
