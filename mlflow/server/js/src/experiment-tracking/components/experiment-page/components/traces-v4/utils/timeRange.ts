/**
 * Time-range primitives for the V4 traces tab. A siloed copy of the shared `useMonitoringFilters`
 * label type + `startTimeLabelToStartEndTime` switch (kept local so the V4 tab depends on zero
 * shared time-range persistence state — matching the "local copy" convention already used by
 * `dateUtils.ts`). The V4 dropdown additionally offers the short "Last 5/15 minutes" presets.
 */

export type TracesV4TimeLabel =
  | 'LAST_5_MINUTES'
  | 'LAST_15_MINUTES'
  | 'LAST_HOUR'
  | 'LAST_24_HOURS'
  | 'LAST_7_DAYS'
  | 'LAST_30_DAYS'
  | 'LAST_YEAR'
  | 'ALL'
  | 'CUSTOM';

/** V4 defaults to a week so users can see recent activity without first widening the time range. */
export const DEFAULT_TRACES_V4_TIME_LABEL: TracesV4TimeLabel = 'LAST_7_DAYS';

const VALID_TRACES_V4_TIME_LABELS = new Set<TracesV4TimeLabel>([
  'LAST_5_MINUTES',
  'LAST_15_MINUTES',
  'LAST_HOUR',
  'LAST_24_HOURS',
  'LAST_7_DAYS',
  'LAST_30_DAYS',
  'LAST_YEAR',
  'ALL',
  'CUSTOM',
]);

const isValidTimeLabel = (value: string): value is TracesV4TimeLabel =>
  VALID_TRACES_V4_TIME_LABELS.has(value as TracesV4TimeLabel);

/** Narrow an untrusted string (URL param / localStorage) to a valid label, or `undefined`. */
export const toValidTimeLabel = (value: string | null | undefined): TracesV4TimeLabel | undefined =>
  value && isValidTimeLabel(value) ? value : undefined;

export interface StartEndTime {
  startTime?: string;
  endTime?: string;
}

/**
 * Resolve a relative label to an ISO start/end range anchored at `dateNow`. `CUSTOM` carries its own
 * explicit bounds elsewhere, so it returns an empty range here. Siloed copy of the shared
 * `startTimeLabelToStartEndTime` (incl. the 5/15-minute cases).
 */
export const getStartEndForLabel = (dateNow: Date, label: TracesV4TimeLabel): StartEndTime => {
  switch (label) {
    case 'LAST_5_MINUTES':
      return {
        startTime: new Date(new Date(dateNow).setUTCMinutes(new Date().getUTCMinutes() - 5)).toISOString(),
        endTime: dateNow.toISOString(),
      };
    case 'LAST_15_MINUTES':
      return {
        startTime: new Date(new Date(dateNow).setUTCMinutes(new Date().getUTCMinutes() - 15)).toISOString(),
        endTime: dateNow.toISOString(),
      };
    case 'LAST_HOUR':
      return {
        startTime: new Date(new Date(dateNow).setUTCHours(new Date().getUTCHours() - 1)).toISOString(),
        endTime: dateNow.toISOString(),
      };
    case 'LAST_24_HOURS':
      return {
        startTime: new Date(new Date(dateNow).setUTCDate(new Date().getUTCDate() - 1)).toISOString(),
        endTime: dateNow.toISOString(),
      };
    case 'LAST_7_DAYS':
      return {
        startTime: new Date(new Date(dateNow).setUTCDate(new Date().getUTCDate() - 7)).toISOString(),
        endTime: dateNow.toISOString(),
      };
    case 'LAST_30_DAYS':
      return {
        startTime: new Date(new Date(dateNow).setUTCDate(new Date().getUTCDate() - 30)).toISOString(),
        endTime: dateNow.toISOString(),
      };
    case 'LAST_YEAR':
      return {
        startTime: new Date(new Date(dateNow).setUTCFullYear(new Date().getUTCFullYear() - 1)).toISOString(),
        endTime: dateNow.toISOString(),
      };
    case 'ALL':
      return { startTime: undefined, endTime: dateNow.toISOString() };
    case 'CUSTOM':
      return { startTime: undefined, endTime: undefined };
    default:
      return { startTime: undefined, endTime: undefined };
  }
};
