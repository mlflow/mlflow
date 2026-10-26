import { describe, expect, test } from '@jest/globals';

import { MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE_TAG, MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG } from '../../constants';
import {
  DEFAULT_ISSUE_DETECTION_START_TIME,
  DEFAULT_ISSUE_DETECTION_WEEKDAY,
  ISSUE_DETECTION_SCHEDULE_VERSION,
  START_TIME_STEP_MINUTES,
  joinStartTime,
  parseIssueDetectionStartTime,
  parseIssueDetectionWeekday,
  parseScheduleTraceFilters,
  readIssueDetectionScheduleFromTags,
  scheduleTiming,
  splitStartTime,
  startTimeHourOptions,
  startTimeMinuteOptions,
  unsupportedFilterCharacter,
} from './issueDetectionSchedule';

const clause = (raw: unknown) =>
  parseScheduleTraceFilters(typeof raw === 'string' ? raw : JSON.stringify(raw)).map(
    (f) => `${f.field === 'metadata' ? 'request_metadata' : 'tags'}.\`${f.key}\` ${f.operator} '${f.value}'`,
  );

describe('parseScheduleTraceFilters', () => {
  test('a well-formed clause of each namespace is kept', () => {
    expect(clause([{ field: 'metadata', operator: '=', key: 'environment', value: 'production' }])).toEqual([
      "request_metadata.`environment` = 'production'",
    ]);
    expect(clause([{ field: 'tag', operator: '=', key: 'team', value: 'search' }])).toEqual(["tags.`team` = 'search'"]);
  });

  test('several clauses are kept in order, to be ANDed', () => {
    expect(
      clause([
        { field: 'metadata', key: 'env', value: 'prod' },
        { field: 'tag', key: 'team', value: 'search' },
      ]),
    ).toEqual(["request_metadata.`env` = 'prod'", "tags.`team` = 'search'"]);
  });

  test.each([
    ['a missing namespace is request metadata', { key: 'env', value: 'prod' }, "request_metadata.`env` = 'prod'"],
    [
      'an unrecognized namespace is request metadata',
      { field: 'span', key: 'env', value: 'prod' },
      "request_metadata.`env` = 'prod'",
    ],
    ['a missing operator is equality', { field: 'tag', key: 'team', value: 'search' }, "tags.`team` = 'search'"],
    ['a padded key is trimmed', { field: 'tag', key: '  team  ', value: 'search' }, "tags.`team` = 'search'"],
    ['a missing value is empty', { field: 'tag', key: 'team' }, "tags.`team` = ''"],
    ['a non-text value is empty', { field: 'tag', key: 'team', value: 7 }, "tags.`team` = ''"],
  ])('%s', (_label, entry, expected) => {
    // Each of these is kept rather than dropped, because dropping a clause widens what the
    // scheduled run analyzes -- the opposite of what the clause was written to do.
    expect(clause([entry])).toEqual([expected]);
  });

  test.each([
    ['a blank key', { field: 'tag', key: '   ', value: 'search' }],
    ['a non-text key', { field: 'tag', key: 7, value: 'search' }],
    ['an operator this version cannot apply', { field: 'tag', operator: '!=', key: 'team', value: 'search' }],
    ['a non-object entry', 'not an object'],
  ])('%s drops the clause', (_label, entry) => {
    expect(clause([entry])).toEqual([]);
  });

  test.each([
    ['an absent tag', ''],
    ['an empty array', '[]'],
    ['an object rather than an array', '{"field":"tag"}'],
    ['junk', 'not-json'],
  ])('%s reads as no filters', (_label, raw) => {
    expect(clause(raw)).toEqual([]);
  });
});

describe('unsupportedFilterCharacter', () => {
  const clauseWith = (value: string) => ({ field: 'metadata' as const, operator: '=' as const, key: 'env', value });

  test.each([
    ['a backtick', '`', '`'],
    ['a single quote', "o'brien", "'"],
    ['a double quote', 'say "yes"', '"'],
    ['a backslash', 'a\\b', '\\'],
  ])('%s is reported as itself', (_label, value, expected) => {
    expect(unsupportedFilterCharacter(clauseWith(value))).toBe(expected);
  });

  test.each([
    ['a tab', 'production\t', 'U+0009'],
    ['a newline', 'production\n', 'U+000A'],
    ['a NUL', 'production\u0000', 'U+0000'],
    ['a DEL', 'production\u007f', 'U+007F'],
  ])('%s is reported by code point, since it shows nothing in the field', (_label, value, expected) => {
    // The server drops a clause carrying one of these, so a schedule saved with it would show a
    // filter in the modal that no run ever applies.
    expect(unsupportedFilterCharacter(clauseWith(value))).toBe(expected);
  });

  test('a control character in the key is caught too', () => {
    expect(unsupportedFilterCharacter({ field: 'tag', operator: '=', key: 'te\tam', value: 'search' })).toBe('U+0009');
  });

  test('an ordinary value is accepted, spaces and all', () => {
    expect(unsupportedFilterCharacter(clauseWith('north america'))).toBeUndefined();
  });
});

describe('parseIssueDetectionStartTime', () => {
  test.each([
    ['a quarter hour the modal offers', '14:30', '14:30'],
    ['midnight', '00:00', '00:00'],
    ['an unpadded time', '9:5', '09:05'],
    ['a padded string', '  09:15  ', '09:15'],
    // The modal offers quarter hours only, but a tag can name any minute. Keeping it is what
    // makes an off-grid time fire at the first poll after it instead of a quarter hour early.
    ['a minute off the poll grid', '09:08', '09:08'],
  ])('%s reads as %s', (_label, raw, expected) => {
    expect(parseIssueDetectionStartTime(raw)).toBe(expected);
  });

  test.each([
    ['an hour past the day', '24:00'],
    ['a minute past the hour', '09:60'],
    ['a negative hour', '-1:30'],
    ['no separator', '0900'],
    ['letters', 'nine'],
    ['seconds as well', '09:00:00'],
    // Readable as 02:00 by the server's `int`, but both readers take ASCII digits only -- a value
    // one of them accepted alone would show one hour in the modal and fire at another.
    ['digits outside ASCII', '\u0662:\u0660\u0660'],
    ['a number rather than a string', 900],
    ['nothing stored', undefined],
  ])('%s falls back to the default', (_label, raw) => {
    // Falling back costs the owner the hour they chose rather than every run, which is the same
    // trade-off the server makes reading the same tag.
    expect(parseIssueDetectionStartTime(raw)).toBe(DEFAULT_ISSUE_DETECTION_START_TIME);
  });
});

describe('the start time the two selects edit', () => {
  test('offers every hour of the day, zero-padded', () => {
    const hours = startTimeHourOptions();
    expect(hours).toHaveLength(24);
    expect(hours.slice(0, 3)).toEqual(['00', '01', '02']);
    expect(hours.at(-1)).toBe('23');
  });

  test('offers only the minutes the dispatcher can fire at', () => {
    // It polls on this grid, so a minute between two of these could not run at the minute it
    // names -- it would wait for the next poll.
    const minutes = startTimeMinuteOptions();
    expect(minutes).toEqual(['00', '15', '30', '45']);
    expect(minutes.every((minute) => Number(minute) % START_TIME_STEP_MINUTES === 0)).toBe(true);
  });

  test('splits and rejoins a stored time without changing it', () => {
    const { hour, minute } = splitStartTime('06:45');
    expect([hour, minute]).toEqual(['06', '45']);
    expect(joinStartTime(hour, minute)).toBe('06:45');
  });

  test('rejoining an off-grid minute keeps it, so editing only the hour cannot round it', () => {
    const { minute } = splitStartTime('09:08');
    expect(joinStartTime('14', minute)).toBe('14:08');
  });
});

describe('parseIssueDetectionWeekday', () => {
  test.each([
    ['Monday', 0, 0],
    ['Sunday', 6, 6],
  ])('%s reads as %i', (_label, raw, expected) => {
    expect(parseIssueDetectionWeekday(raw)).toBe(expected);
  });

  test.each([
    ['a day past the week', 7],
    ['a negative day', -1],
    ['a fraction', 1.5],
    ['a name', 'monday'],
    // `true` is 1 under `==`, and reading it as Tuesday would move a schedule a day because a
    // tag held a boolean. The server refuses it for the same reason.
    ['a boolean', true],
    ['nothing stored', undefined],
  ])('%s falls back to Monday', (_label, raw) => {
    expect(parseIssueDetectionWeekday(raw)).toBe(DEFAULT_ISSUE_DETECTION_WEEKDAY);
  });
});

describe('readIssueDetectionScheduleFromTags', () => {
  const read = (stored: Record<string, unknown>) =>
    readIssueDetectionScheduleFromTags([
      {
        key: MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG,
        value: JSON.stringify({ version: ISSUE_DETECTION_SCHEDULE_VERSION, ...stored }),
      },
    ]);

  test('reads the stored start time', () => {
    expect(read({ cadence: 'daily', fireTime: '06:45' }).startTime).toBe('06:45');
  });

  test('reads the stored weekday', () => {
    expect(read({ cadence: 'weekly', weekday: 4 }).weekday).toBe(4);
  });

  test('a schedule storing no weekday reads as Monday, the day it was already firing on', () => {
    expect(read({ cadence: 'weekly' }).weekday).toBe(DEFAULT_ISSUE_DETECTION_WEEKDAY);
  });

  test('a schedule storing no start time reads as the hour it was already firing at', () => {
    // The compatibility property: a schedule saved before a start time could be chosen must not
    // appear to move when this control shows it.
    expect(read({ cadence: 'daily' }).startTime).toBe(DEFAULT_ISSUE_DETECTION_START_TIME);
  });

  test('a schedule with no readable envelope still reads as a usable start time', () => {
    expect(readIssueDetectionScheduleFromTags([]).startTime).toBe(DEFAULT_ISSUE_DETECTION_START_TIME);
  });
});

describe('the next run the server has worked out', () => {
  const NEXT_RUN_MS = 1_800_000_000_000;

  const readWithState = (stored: Record<string, unknown>, state: Record<string, unknown>) =>
    readIssueDetectionScheduleFromTags([
      {
        key: MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG,
        value: JSON.stringify({ version: ISSUE_DETECTION_SCHEDULE_VERSION, ...stored }),
      },
      {
        key: MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE_TAG,
        value: JSON.stringify({ version: ISSUE_DETECTION_SCHEDULE_VERSION, lastFireTimeMs: 1, ...state }),
      },
    ]);

  const daily = { cadence: 'daily', fireTime: '16:00', weekday: 0, timezone: 'Europe/Berlin' };
  const matchingTiming = scheduleTiming('daily', '16:00', 0, 'Europe/Berlin');

  test('is read when it was recorded against the schedule as it stands', () => {
    expect(readWithState(daily, { timing: matchingTiming, nextFireTimeMs: NEXT_RUN_MS }).nextRunMs).toBe(NEXT_RUN_MS);
  });

  test('matches the fingerprint the server writes', () => {
    // Pinned as a literal as well as through the helper: the two sides have to agree on this
    // string, and a silent change here would only ever hide the line, never be caught by types.
    expect(matchingTiming).toBe('daily|16:00|0|Europe/Berlin');
  });

  test.each([
    ['the start time has changed since', scheduleTiming('daily', '09:00', 0, 'Europe/Berlin')],
    ['the cadence has changed since', scheduleTiming('weekly', '16:00', 0, 'Europe/Berlin')],
    ['the weekday has changed since', scheduleTiming('daily', '16:00', 3, 'Europe/Berlin')],
    ['the timezone has changed since', scheduleTiming('daily', '16:00', 0, 'UTC')],
  ])('is withheld when %s', (_label, timing) => {
    // Showing it would name a time nothing will happen at: the server has not recomputed the
    // next run for the schedule that is stored now.
    expect(readWithState(daily, { timing, nextFireTimeMs: NEXT_RUN_MS }).nextRunMs).toBeUndefined();
  });

  test.each([
    ['no timing recorded', { nextFireTimeMs: NEXT_RUN_MS }],
    ['no next run recorded', { timing: matchingTiming }],
    ['a next run that is not a number', { timing: matchingTiming, nextFireTimeMs: 'soon' }],
  ])('is withheld when there is %s', (_label, state) => {
    expect(readWithState(daily, state).nextRunMs).toBeUndefined();
  });

  test('is withheld when there is no state tag at all', () => {
    expect(
      readIssueDetectionScheduleFromTags([
        {
          key: MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG,
          value: JSON.stringify({ version: ISSUE_DETECTION_SCHEDULE_VERSION, ...daily }),
        },
      ]).nextRunMs,
    ).toBeUndefined();
  });
});
