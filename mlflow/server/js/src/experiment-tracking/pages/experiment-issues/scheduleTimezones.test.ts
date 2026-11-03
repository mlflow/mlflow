import { describe, expect, test } from '@jest/globals';

import { allTimezones, browserTimezone, recommendedTimezones, timezoneChoice } from './scheduleTimezones';

describe('timezoneChoice', () => {
  test.each([
    ['UTC', '(UTC+00:00) UTC'],
    // A whole-hour offset and a half-hour one, since the label is what tells an owner when their
    // chosen time actually falls.
    ['Asia/Tokyo', '(UTC+09:00) Asia/Tokyo'],
    // The half-hour offset matters as much as the whole-hour ones: it is what tells an owner in
    // India that 09:00 there is not 09:00 anywhere else.
    ['Asia/Calcutta', '(UTC+05:30) Asia/Calcutta'],
  ])('labels %s as %s', (zone, expected) => {
    expect(timezoneChoice(zone).label).toBe(expected);
  });

  test('a zone west of UTC is signed', () => {
    // Offsets shift with daylight saving, so the sign and the name are asserted rather than the
    // exact hour, which would make this test fail twice a year.
    const { label, offsetMinutes } = timezoneChoice('America/Los_Angeles');
    expect(offsetMinutes).toBeLessThan(0);
    expect(label).toMatch(/^\(UTC-0[78]:00\) America\/Los_Angeles$/);
  });

  test('a zone the browser cannot format still produces a choice', () => {
    // Falling back keeps an unknown zone selectable and labelled rather than dropping it, which
    // matters because the server, not the browser, decides which zones it can load.
    expect(timezoneChoice('Mars/Olympus_Mons')).toEqual({
      zone: 'Mars/Olympus_Mons',
      label: '(UTC+00:00) Mars/Olympus_Mons',
      offsetMinutes: 0,
    });
  });
});

describe('allTimezones', () => {
  test('offers the real zone list, west to east', () => {
    const zones = allTimezones();
    expect(zones.length).toBeGreaterThan(100);
    const offsets = zones.map(({ offsetMinutes }) => offsetMinutes);
    expect(offsets).toEqual([...offsets].sort((a, b) => a - b));
    expect(zones.map(({ zone }) => zone)).toContain('Asia/Tokyo');
  });

  test('names no zone twice', () => {
    const zones = allTimezones().map(({ zone }) => zone);
    expect(new Set(zones).size).toBe(zones.length);
  });

  test('carries whatever ids the platform calls canonical, legacy names included', () => {
    // `Intl` keeps the older canonical id for several zones -- India is `Asia/Calcutta`, not
    // `Asia/Kolkata` -- so searching the modern name finds nothing. Pinned because it is
    // surprising rather than because it is wanted; both ids load on the server, so a schedule
    // saved with either fires at the right time.
    const zones = allTimezones().map(({ zone }) => zone);
    expect(zones).toContain('Asia/Calcutta');
    expect(zones).not.toContain('Asia/Kolkata');
  });

  test('UTC is offered even though the platform list leaves it out', () => {
    // `supportedValuesOf` lists `Etc/UTC` and friends but not a bare `UTC`, which is the one
    // people reach for, so it is suggested explicitly.
    expect(recommendedTimezones().map(({ zone }) => zone)).toContain('UTC');
  });
});

describe('recommendedTimezones', () => {
  test('always offers UTC and the zone this browser is in', () => {
    const zones = recommendedTimezones().map(({ zone }) => zone);
    expect(zones).toContain('UTC');
    expect(zones).toContain(browserTimezone());
  });

  test("offers the schedule's own zone, so re-opening the dialog does not bury it", () => {
    expect(recommendedTimezones('Asia/Tokyo').map(({ zone }) => zone)).toContain('Asia/Tokyo');
  });

  test('does not repeat a zone that is already suggested', () => {
    const zones = recommendedTimezones('UTC').map(({ zone }) => zone);
    expect(new Set(zones).size).toBe(zones.length);
  });
});
