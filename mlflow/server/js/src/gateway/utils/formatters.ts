/**
 * Format token count for display (e.g., 128000 -> "128K")
 * Returns null for null/undefined inputs to allow conditional rendering
 */
export const formatTokens = (tokens: number | null | undefined): string | null => {
  if (tokens === null || tokens === undefined) return null;
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M`;
  if (tokens >= 1_000) return `${(tokens / 1_000).toFixed(0)}K`;
  return tokens.toString();
};

/**
 * Format cost per token for display (e.g., 0.000002 -> "$2.00/1M")
 * Converts per-token cost to cost per million tokens for readability
 * Returns null for null/undefined inputs to allow conditional rendering
 */
export const formatCost = (cost: number | null | undefined): string | null => {
  if (cost === null || cost === undefined) return null;
  if (cost === 0) return 'Free';
  const perMillion = cost * 1_000_000;
  if (perMillion < 0.01) return `$${perMillion.toFixed(4)}/1M`;
  return `$${perMillion.toFixed(2)}/1M`;
};
