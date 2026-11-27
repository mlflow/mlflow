const usdFormatters = new Map<number, Intl.NumberFormat>();

/**
 * Formats a cost value in USD with appropriate precision.
 * Shows 2-6 decimal places depending on the value by default.
 */
export const formatCostUSD = (cost: number, maximumFractionDigits = 6): string => {
  let formatter = usdFormatters.get(maximumFractionDigits);
  if (!formatter) {
    formatter = new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits,
    });
    usdFormatters.set(maximumFractionDigits, formatter);
  }
  return formatter.format(cost);
};

/** ≈ prefix for all pricing-table-estimated cost values — defined once to keep the glyph out of call sites. */
export const ESTIMATED_COST_PREFIX = '≈';

/** Formats a cost as an estimated USD amount, prefixed with ≈ to signal it is computed from token usage, not provider-reported. */
export const formatEstimatedCostUSD = (cost: number, maximumFractionDigits = 6): string =>
  `${ESTIMATED_COST_PREFIX}${formatCostUSD(cost, maximumFractionDigits)}`;
