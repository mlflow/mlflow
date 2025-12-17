/**
 * Converts a Unix timestamp (in seconds) to a Date object.
 * Gateway API returns timestamps in seconds, so we multiply by 1000 to get milliseconds.
 */
export const timestampToDate = (timestamp: number | undefined | null): Date => {
  if (timestamp === undefined || timestamp === null) {
    return new Date();
  }

  // Gateway API returns Unix timestamps in seconds - convert to milliseconds
  // A timestamp less than 10 billion is definitely in seconds (before year 2286)
  // A timestamp greater than 1 trillion is definitely in milliseconds (after 2001)
  if (timestamp < 10_000_000_000) {
    return new Date(timestamp * 1000);
  }

  return new Date(timestamp);
};
