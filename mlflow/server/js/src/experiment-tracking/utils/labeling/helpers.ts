/**
 * Helper utilities for labeling functionality
 *
 * This file contains general-purpose utilities for labeling sessions and schemas.
 * Agent interaction helpers have been excluded from the OSS port.
 *
 * Additional helpers can be added here as needed by labeling UI components.
 */

/**
 * Checks if a value is not null or undefined
 */
export function hasValue<T>(value: T | null | undefined): value is T {
  return value !== null && value !== undefined;
}

/**
 * Format a timestamp to a human-readable date string
 */
export function formatTimestamp(timestampMs: number): string {
  return new Date(timestampMs).toLocaleString();
}

/**
 * Generate a unique ID for labeling items
 */
export function generateLabelingItemId(): string {
  return `labeling_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}
