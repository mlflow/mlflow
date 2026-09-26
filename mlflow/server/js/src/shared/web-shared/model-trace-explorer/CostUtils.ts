/**
 * Formats a cost value in USD with appropriate precision.
 * Shows 2-6 decimal places depending on the value.
 */
export const formatCostUSD = (cost: number, maximumFractionDigits = 6): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits,
  }).format(cost);
};
