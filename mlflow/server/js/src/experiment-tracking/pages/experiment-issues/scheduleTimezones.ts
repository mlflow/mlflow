/**
 * The timezone choices the schedule modal offers, labelled by their current UTC offset.
 *
 * A schedule's cadence fires at a wall-clock time, so the zone that time is read in is part of
 * what the owner is choosing. The label carries the offset because a zone name alone does not say
 * when it fires -- `Asia/Tokyo` means nothing to someone deciding between it and their own zone.
 */

/** Suggested first: UTC, and the zone the editor's own browser is in. */
export const RECOMMENDED_TIMEZONES = ['UTC'] as const;

export interface TimezoneChoice {
  /** IANA zone id, which is what the schedule stores. */
  zone: string;
  /** `(UTC+09:00) Asia/Tokyo`, the form shown in the list and in the closed control. */
  label: string;
  /** Minutes east of UTC, for sorting. */
  offsetMinutes: number;
}

/**
 * The zone's current offset in minutes east of UTC.
 *
 * Read from `Intl` rather than from a table, so it follows the browser's own tzdata and reflects
 * daylight saving as it stands today. It is only a label: the schedule stores the zone, and the
 * server resolves the offset again at each fire time, so an offset that changes later cannot move
 * a schedule.
 */
const offsetMinutesOf = (zone: string): number => {
  try {
    const formatted = new Intl.DateTimeFormat('en-US', { timeZone: zone, timeZoneName: 'longOffset' })
      .formatToParts(new Date())
      .find((part) => part.type === 'timeZoneName')?.value;
    // `GMT+09:00`, or a bare `GMT` at zero offset.
    const match = /GMT([+-])(\d{2}):(\d{2})/.exec(formatted ?? '');
    if (!match) {
      return 0;
    }
    const minutes = Number(match[2]) * 60 + Number(match[3]);
    return match[1] === '-' ? -minutes : minutes;
  } catch {
    // A zone the browser cannot format sorts with UTC rather than dropping out of the list.
    return 0;
  }
};

const formatOffset = (offsetMinutes: number): string => {
  const sign = offsetMinutes < 0 ? '-' : '+';
  const absolute = Math.abs(offsetMinutes);
  const pad = (value: number) => String(value).padStart(2, '0');
  return `${sign}${pad(Math.floor(absolute / 60))}:${pad(absolute % 60)}`;
};

export const timezoneChoice = (zone: string): TimezoneChoice => {
  const offsetMinutes = offsetMinutesOf(zone);
  return { zone, label: `(UTC${formatOffset(offsetMinutes)}) ${zone}`, offsetMinutes };
};

/** The editor's own zone, or `UTC` when the browser will not say. */
export const browserTimezone = (): string => Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';

/**
 * Every zone this browser knows, west to east, then alphabetically within an offset.
 *
 * `Intl.supportedValuesOf` is the only list available to a client, and a browser without it still
 * has to be able to pick a zone -- so it falls back to the two that are always meaningful: UTC and
 * the one the editor is in.
 */
export const allTimezones = (): TimezoneChoice[] => {
  let zones: readonly string[];
  try {
    zones = Intl.supportedValuesOf('timeZone');
  } catch {
    zones = [...RECOMMENDED_TIMEZONES, browserTimezone()];
  }
  return [...new Set(zones)]
    .map(timezoneChoice)
    .sort((a, b) => a.offsetMinutes - b.offsetMinutes || a.zone.localeCompare(b.zone));
};

/**
 * UTC and the editor's zone, in that order and without repeating one.
 *
 * Offered separately because they are the two answers that need no thought: the zone the editor is
 * reading the dialog in, and the one that never shifts.
 */
export const recommendedTimezones = (stored?: string): TimezoneChoice[] => {
  const zones = [...RECOMMENDED_TIMEZONES, browserTimezone()];
  // A zone already on the schedule belongs here too, so re-opening the dialog does not bury the
  // schedule's own zone several hundred entries down a list.
  if (stored) {
    zones.push(stored);
  }
  return [...new Set(zones)].map(timezoneChoice);
};
