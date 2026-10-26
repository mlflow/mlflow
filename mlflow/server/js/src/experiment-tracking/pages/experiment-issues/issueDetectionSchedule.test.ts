import { describe, expect, test } from '@jest/globals';

import { parseScheduleTraceFilters, unsupportedFilterCharacter } from './issueDetectionSchedule';

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
