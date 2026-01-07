/**
 * Utility functions for processing and bucketing distribution data.
 * Used for creating histogram-style visualizations from raw data values.
 */

export interface HistogramBucket {
  min: number;
  max: number;
  label: string;
}

/**
 * Sort values intelligently using alphanumeric comparison.
 * Numbers are sorted numerically, strings alphabetically.
 */
export const sortValuesAlphanumerically = (values: string[]): string[] =>
  [...values].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));

/**
 * Determine if numeric values should be bucketed into histogram ranges.
 * Returns true if:
 * - Values contain floats (have decimals), OR
 * - Values are integers with more than the specified threshold of unique values
 *
 * Returns false for non-numeric values (strings, booleans) or sparse integer data.
 *
 * @param values - Array of string values to analyze
 * @param uniqueThreshold - Maximum unique integer values before bucketing (default: 5)
 * @returns true if values should be bucketed into ranges
 */
export const shouldCreateHistogramBuckets = (values: string[], uniqueThreshold = 5): boolean => {
  if (values.length === 0) return false;

  const numericValues = values.map((v) => parseFloat(v)).filter((n) => !isNaN(n));
  const uniqueValues = new Set(numericValues);
  const hasDecimals = numericValues.some((n) => !Number.isInteger(n));

  // Bucket if: values are floats, OR integers with more than threshold unique values
  return hasDecimals || uniqueValues.size > uniqueThreshold;
};

/**
 * Create histogram buckets for continuous numeric values.
 * Divides the range [min, max] into equal-sized buckets.
 *
 * @param values - Array of string values to bucket
 * @param numBuckets - Number of buckets to create (default: 5)
 * @returns Array of bucket definitions with min, max, and label
 */
export const createHistogramBuckets = (values: string[], numBuckets = 5): HistogramBucket[] => {
  const numericValues = values.map((v) => parseFloat(v)).filter((n) => !isNaN(n));
  if (numericValues.length === 0) return [];

  const min = Math.min(...numericValues);
  const max = Math.max(...numericValues);

  // Handle edge case where all values are identical
  if (min === max) {
    return [{ min, max, label: min.toFixed(2) }];
  }

  const range = max - min;
  const bucketSize = range / numBuckets;

  return Array.from({ length: numBuckets }, (_, i) => {
    const bucketMin = min + i * bucketSize;
    const bucketMax = i === numBuckets - 1 ? max : min + (i + 1) * bucketSize;
    return {
      min: bucketMin,
      max: bucketMax,
      label: `${bucketMin.toFixed(2)}-${bucketMax.toFixed(2)}`,
    };
  });
};

/**
 * Find the bucket index for a given numeric value.
 * Uses [min, max) for all buckets except the last one which uses [min, max].
 *
 * @param value - The numeric value to place in a bucket
 * @param buckets - Array of bucket definitions
 * @returns The index of the bucket containing the value
 */
export const findBucketIndexForValue = (value: number, buckets: HistogramBucket[]): number => {
  for (let i = 0; i < buckets.length; i++) {
    const isLastBucket = i === buckets.length - 1;
    // Use [min, max) for all buckets except the last one which is [min, max]
    if (value >= buckets[i].min && (isLastBucket ? value <= buckets[i].max : value < buckets[i].max)) {
      return i;
    }
  }
  return buckets.length - 1; // Default to last bucket
};
