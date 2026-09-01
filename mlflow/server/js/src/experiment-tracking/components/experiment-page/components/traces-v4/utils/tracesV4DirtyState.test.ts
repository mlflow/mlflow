import { describe, expect, test } from '@jest/globals';
import { capturedV4StatesMatch, __test__ } from './tracesV4DirtyState';
import { captureV4ViewState } from './tracesV4SavedViewState';
import { FilterOp, type TraceColumnId, type TraceFilterModel } from '@databricks/web-shared/traces-table';

const { canonicalViewQuery, columnSetsEqual, assessmentVisibilityEqual, customVisibilityEqual } = __test__;

const capture = (
  query: string,
  cols: TraceColumnId[] = [],
  filters: TraceFilterModel = [],
  assessments: Record<string, boolean> = {},
  custom: Record<string, boolean> = {},
) => captureV4ViewState(new URLSearchParams(query), cols, filters, assessments, custom);

describe('capturedV4StatesMatch', () => {
  test('identical captures match (clean)', () => {
    const a = capture('q=x&sort=duration&dir=asc', ['start_time', 'input']);
    const b = capture('q=x&sort=duration&dir=asc', ['start_time', 'input']);
    expect(capturedV4StatesMatch(a, b)).toBe(true);
  });

  test('a different search query is dirty', () => {
    const a = capture('q=changed', ['start_time']);
    const b = capture('q=x', ['start_time']);
    expect(capturedV4StatesMatch(a, b)).toBe(false);
  });

  test('columns are compared as a set — reordering is still clean', () => {
    const a = capture('q=x', ['input', 'start_time']);
    const b = capture('q=x', ['start_time', 'input']);
    expect(capturedV4StatesMatch(a, b)).toBe(true);
  });

  test('a different column selection is dirty', () => {
    const a = capture('q=x', ['start_time', 'input', 'duration']);
    const b = capture('q=x', ['start_time', 'input']);
    expect(capturedV4StatesMatch(a, b)).toBe(false);
  });

  test('tag filters must match (including order)', () => {
    const a = capture('tag=env%3Dprod&tag=team%3Dml');
    const b = capture('tag=env%3Dprod&tag=team%3Dml');
    expect(capturedV4StatesMatch(a, b)).toBe(true);
    const c = capture('tag=team%3Dml&tag=env%3Dprod');
    expect(capturedV4StatesMatch(a, c)).toBe(false);
  });

  test('an absent filter model matches an empty one (legacy view stays clean)', () => {
    const a = capture('q=x', ['start_time']); // no filters → undefined
    const b = capture('q=x', ['start_time'], []); // empty filters → omitted too
    expect(capturedV4StatesMatch(a, b)).toBe(true);
  });

  test('the popover filter model is part of the diff', () => {
    const a = capture('q=x', ['start_time'], [{ field: 'state', operator: FilterOp.EQUALS, value: 'ERROR' }]);
    const b = capture('q=x', ['start_time'], []);
    expect(capturedV4StatesMatch(a, b)).toBe(false);
    const c = capture('q=x', ['start_time'], [{ field: 'state', operator: FilterOp.EQUALS, value: 'ERROR' }]);
    expect(capturedV4StatesMatch(a, c)).toBe(true);
  });

  test('toggling session grouping is dirty', () => {
    const grouped = capture('q=x&groupBy=session', ['start_time']);
    const flat = capture('q=x', ['start_time']);
    expect(capturedV4StatesMatch(grouped, flat)).toBe(false);
    expect(capturedV4StatesMatch(grouped, capture('q=x&groupBy=session', ['start_time']))).toBe(true);
  });

  test('hiding an assessment column is dirty', () => {
    const a = capture('q=x', ['start_time'], [], { correctness: true });
    const b = capture('q=x', ['start_time'], [], { correctness: false });
    expect(capturedV4StatesMatch(a, b)).toBe(false);
  });

  test('showing a custom column is dirty', () => {
    const a = capture('q=x', ['start_time'], [], {}, { 'tag:environment': true });
    const b = capture('q=x', ['start_time'], [], {}, { 'tag:environment': false });
    expect(capturedV4StatesMatch(a, b)).toBe(false);
  });

  test('an absent custom column entry is treated as hidden (opt-in default)', () => {
    const a = capture('q=x', ['start_time'], [], {}, {});
    const b = capture('q=x', ['start_time'], [], {}, { 'tag:environment': false });
    expect(capturedV4StatesMatch(a, b)).toBe(true);
  });
});

describe('canonicalViewQuery', () => {
  test('drops the recomputed absolute bounds for a relative time label', () => {
    // Two captures of the same relative range differ only in their recomputed startTime/endTime;
    // they must canonicalize equal so a passive re-render never reads as dirty.
    const a = capture('startTimeLabel=LAST_7_DAYS&startTime=2026-01-01T00:00:00Z&endTime=2026-01-08T00:00:00Z');
    const b = capture('startTimeLabel=LAST_7_DAYS&startTime=2026-02-01T00:00:00Z&endTime=2026-02-08T00:00:00Z');
    expect(canonicalViewQuery(a)).toBe(canonicalViewQuery(b));
  });

  test('keeps the explicit bounds for a CUSTOM range', () => {
    const a = capture('startTimeLabel=CUSTOM&startTime=2026-01-01T00:00:00Z&endTime=2026-01-08T00:00:00Z');
    const b = capture('startTimeLabel=CUSTOM&startTime=2026-02-01T00:00:00Z&endTime=2026-02-08T00:00:00Z');
    expect(canonicalViewQuery(a)).not.toBe(canonicalViewQuery(b));
  });

  test('normalizes the default label so an absent label matches the explicit default', () => {
    // A stored view saved before the default label was written vs a fresh capture that includes it.
    const withDefault = capture('q=x&startTimeLabel=LAST_7_DAYS');
    const withoutLabel = capture('q=x');
    expect(canonicalViewQuery(withDefault)).toBe(canonicalViewQuery(withoutLabel));
  });

  test('is order-independent across params', () => {
    const a = capture('q=x&sort=duration&dir=asc');
    const b = capture('sort=duration&dir=asc&q=x');
    expect(canonicalViewQuery(a)).toBe(canonicalViewQuery(b));
  });
});

describe('columnSetsEqual', () => {
  test('true regardless of order, false on differing membership', () => {
    expect(columnSetsEqual(['a', 'b'], ['b', 'a'])).toBe(true);
    expect(columnSetsEqual(['a'], ['a', 'b'])).toBe(false);
    expect(columnSetsEqual(['a', 'b'], ['a', 'c'])).toBe(false);
  });
});

describe('assessmentVisibilityEqual', () => {
  test('compares effective visibility — an absent name is default-visible', () => {
    expect(assessmentVisibilityEqual({ a: true }, {})).toBe(true);
    expect(assessmentVisibilityEqual({ a: true, b: true }, { a: true })).toBe(true);
    expect(assessmentVisibilityEqual({ a: false }, {})).toBe(false);
    expect(assessmentVisibilityEqual({ a: false }, { a: true })).toBe(false);
  });
});

describe('customVisibilityEqual', () => {
  test('compares effective visibility — an absent id is default-hidden (opt-in)', () => {
    expect(customVisibilityEqual({ a: false }, {})).toBe(true);
    expect(customVisibilityEqual({ a: false, b: false }, { a: false })).toBe(true);
    expect(customVisibilityEqual({ a: true }, {})).toBe(false);
    expect(customVisibilityEqual({ a: true }, { a: false })).toBe(false);
  });
});
