export enum TimeUnit {
  Second = 'second',
  Minute = 'minute',
  Hour = 'hour',
  Day = 'day',
  Month = 'month',
  Year = 'year',
}

// Time intervals in seconds for each unit
export const TIME_UNIT_SECONDS: Record<TimeUnit, number> = {
  [TimeUnit.Second]: 1,
  [TimeUnit.Minute]: 60,
  [TimeUnit.Hour]: 3600,
  [TimeUnit.Day]: 86400,
  [TimeUnit.Month]: 2592000,
  [TimeUnit.Year]: 31536000,
};

// Maximum number of data points allowed for charts rendering
export const MAX_DATA_POINTS = 1000;

/**
 * Calculate the expected number of data points for a given time range and unit
 */
export function calculateExpectedDataPoints(
  startTimeMs: number | undefined,
  endTimeMs: number | undefined,
  timeUnit: TimeUnit,
): number {
  if (!startTimeMs || !endTimeMs) {
    return 0;
  }
  const durationSeconds = (endTimeMs - startTimeMs) / 1000;
  return Math.ceil(durationSeconds / TIME_UNIT_SECONDS[timeUnit]);
}

/**
 * Check if a time unit is valid for the given time range
 */
export function isTimeUnitValid(
  startTimeMs: number | undefined,
  endTimeMs: number | undefined,
  timeUnit: TimeUnit,
): boolean {
  const expectedPoints = calculateExpectedDataPoints(startTimeMs, endTimeMs, timeUnit);
  return expectedPoints <= MAX_DATA_POINTS;
}

/**
 * Calculate the appropriate default time unit based on the time range.
 * - <= 1 hour: minute level
 * - <= 24 hours: hour level
 * - <= 1 month: day level
 * - <= 1 year: month level
 * - > 1 year: year level
 */
export function calculateDefaultTimeUnit(startTimeMs?: number, endTimeMs?: number): TimeUnit {
  if (!startTimeMs || !endTimeMs) {
    return TimeUnit.Day;
  }

  const durationMs = endTimeMs - startTimeMs;
  const durationSeconds = durationMs / 1000;

  if (durationSeconds <= TIME_UNIT_SECONDS[TimeUnit.Hour]) {
    return TimeUnit.Minute;
  } else if (durationSeconds <= TIME_UNIT_SECONDS[TimeUnit.Day]) {
    return TimeUnit.Hour;
  } else if (durationSeconds <= TIME_UNIT_SECONDS[TimeUnit.Month]) {
    return TimeUnit.Day;
  } else if (durationSeconds <= TIME_UNIT_SECONDS[TimeUnit.Year]) {
    return TimeUnit.Month;
  } else {
    return TimeUnit.Year;
  }
}
