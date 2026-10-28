/**
 * Formats a cost value in USD with appropriate precision.
 * Shows 2-6 decimal places depending on the value by default.
 */
export const formatCostUSD = (cost: number, maximumFractionDigits = 6): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits,
  }).format(cost);
};

/** ≈ prefix for all pricing-table-estimated cost values — defined once to keep the glyph out of call sites. */
export const ESTIMATED_COST_PREFIX = '≈';

/** Formats a cost as an estimated USD amount, prefixed with ≈ to signal it is computed from token usage, not provider-reported. */
export const formatEstimatedCostUSD = (cost: number, maximumFractionDigits = 6): string =>
  `${ESTIMATED_COST_PREFIX}${formatCostUSD(cost, maximumFractionDigits)}`;
