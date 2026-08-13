/**
 * Parse a duration string of the form `<number><unit>` to milliseconds. The V4 search API returns
 * `execution_duration` as a proto `google.protobuf.Duration` string — always in seconds, e.g.
 * "0.002s", "32.583s", "3600s" — but the other units are accepted defensively so an unexpected
 * shape can't slip through as NaN. Returns null when the string doesn't match, letting the caller
 * fall back to the raw value rather than dropping information.
 */
const DURATION_PATTERN = /^([\d.]+)\s*(s|ms|us|µs|ns|m|h)$/;

const parseDurationToMs = (duration: string): number | null => {
  const match = duration.trim().match(DURATION_PATTERN);
  if (!match) {
    return null;
  }
  const value = Number.parseFloat(match[1]);
  if (!Number.isFinite(value)) {
    return null;
  }
  switch (match[2]) {
    case 'h':
      return value * 3_600_000;
    case 'm':
      return value * 60_000;
    case 's':
      return value * 1_000;
    case 'ms':
      return value;
    case 'us':
    case 'µs':
      return value / 1_000;
    case 'ns':
      return value / 1_000_000;
    default:
      return null;
  }
};

/**
 * Render a millisecond duration with a single, magnitude-appropriate unit. Mirrors the canonical
 * `Utils.formatDuration`, but crosses from milliseconds to seconds at 1000ms (not 500ms) so
 * anything under a second reads as whole milliseconds — the "ms below 1 second" spec for the
 * traces duration column.
 */
const formatDurationMs = (ms: number): string => {
  if (ms < 1_000) {
    return `${Math.round(ms)}ms`;
  }
  if (ms < 1_000 * 60) {
    return `${(ms / 1_000).toFixed(1)}s`;
  }
  if (ms < 1_000 * 60 * 60) {
    return `${(ms / 1_000 / 60).toFixed(1)}min`;
  }
  if (ms < 1_000 * 60 * 60 * 24) {
    return `${(ms / 1_000 / 60 / 60).toFixed(1)}h`;
  }
  return `${(ms / 1_000 / 60 / 60 / 24).toFixed(1)}d`;
};

/**
 * Human-friendly rendering of a V4 `execution_duration` string (e.g. "32.583s" → "32.6s"). Returns
 * null when the value can't be parsed, so the caller can fall back to showing the raw string.
 */
export const formatTraceDuration = (duration: string): string | null => {
  const ms = parseDurationToMs(duration);
  return ms === null ? null : formatDurationMs(ms);
};
