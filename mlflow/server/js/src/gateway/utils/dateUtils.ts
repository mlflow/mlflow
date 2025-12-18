export const timestampToDate = (timestamp: number | undefined | null): Date => {
  if (timestamp === undefined || timestamp === null) {
    return new Date();
  }

  // Gateway API returns Unix timestamps in seconds - convert to milliseconds
  if (timestamp < 10_000_000_000) {
    return new Date(timestamp * 1000);
  }

  return new Date(timestamp);
};
