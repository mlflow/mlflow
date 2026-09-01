import { describe, expect, test } from '@jest/globals';
import { FilterOp, TRACE_COLUMN_IDS } from '@databricks/web-shared/traces-table';
import {
  captureV4ViewState,
  buildV4ViewQuery,
  decodeViewColumns,
  getTraceV4SavedViewShareUrl,
  getTraceV4SavedViewTagKey,
  getTraceV4SavedViewIdFromTagKey,
  urlHasCapturedV4ViewState,
  TRACE_V4_SAVED_VIEW_TAG_PREFIX,
  TRACE_V4_SHARE_URL_PARAM_KEY,
} from './tracesV4SavedViewState';

const params = (query: string) => new URLSearchParams(query);

describe('tracesV4SavedViewState tag-key helpers', () => {
  test('round-trips a view id through the tag key', () => {
    const key = getTraceV4SavedViewTagKey('abc');
    expect(key).toBe(`${TRACE_V4_SAVED_VIEW_TAG_PREFIX}abc`);
    expect(getTraceV4SavedViewIdFromTagKey(key)).toBe('abc');
  });

  test('a key that is exactly the prefix (no id) is not a saved-view key', () => {
    expect(getTraceV4SavedViewIdFromTagKey(TRACE_V4_SAVED_VIEW_TAG_PREFIX)).toBeNull();
  });

  test('rejects tag keys from other view features so V4 never lists V3 / runs views', () => {
    // V3's sort wire format (`key::type::asc`) is incompatible with V4's `sort`+`dir`, so the V4
    // prefix MUST be distinct — a V3 or runs view must never appear in the V4 list.
    expect(getTraceV4SavedViewIdFromTagKey('mlflow.traceViewState.v1')).toBeNull(); // V3
    expect(getTraceV4SavedViewIdFromTagKey('mlflow.sharedViewState.v1')).toBeNull(); // runs
    expect(getTraceV4SavedViewIdFromTagKey('mlflow.note')).toBeNull();
  });
});

describe('captureV4ViewState', () => {
  test('captures the whitelisted URL view params and drops the transient ones', () => {
    const state = captureV4ViewState(
      params(
        'q=refund&sort=duration&dir=asc&pageSize=50&page=3&traceId=abc123&startTimeLabel=LAST_7_DAYS&groupBy=session',
      ),
      ['start_time', 'input', 'duration'],
    );
    const query = buildV4ViewQuery(state, 'view-1');
    const out = params(query);

    expect(out.get('q')).toBe('refund');
    expect(out.get('sort')).toBe('duration');
    expect(out.get('dir')).toBe('asc');
    expect(out.get('pageSize')).toBe('50');
    expect(out.get('startTimeLabel')).toBe('LAST_7_DAYS');
    // Session grouping is a URL-backed view param, so it round-trips like the rest.
    expect(out.get('groupBy')).toBe('session');
    // Transient state is never part of a saved view.
    expect(out.get('page')).toBeNull();
    expect(out.get('traceId')).toBeNull();
  });

  test('preserves repeatable tag filters in order', () => {
    const state = captureV4ViewState(params('tag=env%3Dprod&tag=team%3Dsearch'), ['start_time']);
    const out = params(buildV4ViewQuery(state, 'view-1'));
    expect(out.getAll('tag')).toEqual(['env=prod', 'team=search']);
  });

  test('captures the live visible columns into the stored state (not the URL query)', () => {
    // Columns live in localStorage, not the URL: they are stored in the envelope's `cols` key and
    // restored into the column store on open, so they must NOT appear in the built query string.
    const state = captureV4ViewState(params('q=x'), ['start_time', 'input', 'duration']);
    expect(state.single.cols).toBe('start_time,input,duration');
    const out = params(buildV4ViewQuery(state, 'view-1'));
    expect(out.get('cols')).toBeNull();
  });

  test('stores the live popover filter model, and omits an empty one', () => {
    // Filters are React state (not URL-backed), so they're passed in and stored on the envelope; an
    // empty model is omitted so a filter-less view stays byte-identical to a legacy (pre-filter) one.
    const withFilters = captureV4ViewState(
      params('q=x'),
      ['start_time'],
      [{ field: 'state', operator: FilterOp.EQUALS, value: 'ERROR' }],
    );
    expect(withFilters.filters).toEqual([{ field: 'state', operator: FilterOp.EQUALS, value: 'ERROR' }]);
    // Filters never leak into the rebuilt URL query.
    expect(params(buildV4ViewQuery(withFilters, 'view-1')).has('filters')).toBe(false);

    const noFilters = captureV4ViewState(params('q=x'), ['start_time'], []);
    expect(noFilters.filters).toBeUndefined();
  });

  test('captures assessment-column visibility, omitting an empty map', () => {
    // An empty map is omitted so a view saved with no assessments stays byte-identical to a legacy one.
    const withAssessments = captureV4ViewState(params('q=x'), ['start_time'], [], {
      correctness: true,
      relevance: false,
    });
    expect(withAssessments.assessmentColumns).toEqual({ correctness: true, relevance: false });
    expect(params(buildV4ViewQuery(withAssessments, 'view-1')).has('assessmentColumns')).toBe(false);

    const noAssessments = captureV4ViewState(params('q=x'), ['start_time'], [], {});
    expect(noAssessments.assessmentColumns).toBeUndefined();
  });

  test('captures custom-column visibility, omitting an empty map', () => {
    // An empty map is omitted so a view saved with no custom columns stays byte-identical to a legacy one.
    const withCustom = captureV4ViewState(
      params('q=x'),
      ['start_time'],
      [],
      {},
      {
        'tag:environment': true,
        'custom_metadata:region': false,
      },
    );
    expect(withCustom.customColumns).toEqual({ 'tag:environment': true, 'custom_metadata:region': false });
    expect(params(buildV4ViewQuery(withCustom, 'view-1')).has('customColumns')).toBe(false);

    const noCustom = captureV4ViewState(params('q=x'), ['start_time'], [], {}, {});
    expect(noCustom.customColumns).toBeUndefined();
  });

  test('never captures an incoming share key or cols param from the URL (only the live columns)', () => {
    // Opening view A then saving must not leak A's share key / cols into view B.
    const state = captureV4ViewState(params(`q=x&cols=tokens,cost&${TRACE_V4_SHARE_URL_PARAM_KEY}=other-view`), [
      'start_time',
      'input',
    ]);
    expect(state.single.cols).toBe('start_time,input'); // live columns win, not the URL's cols
    const out = params(buildV4ViewQuery(state, 'view-b'));
    expect(out.get(TRACE_V4_SHARE_URL_PARAM_KEY)).toBe('view-b'); // the new id, not the old one
  });
});

describe('buildV4ViewQuery', () => {
  test('always sets the share key to the given view id', () => {
    const state = captureV4ViewState(params('q=x'), ['start_time']);
    const out = params(buildV4ViewQuery(state, 'my-id'));
    expect(out.get(TRACE_V4_SHARE_URL_PARAM_KEY)).toBe('my-id');
  });
});

describe('urlHasCapturedV4ViewState', () => {
  test('true when the URL carries any serialized view state', () => {
    expect(urlHasCapturedV4ViewState(params('sort=duration'))).toBe(true);
    expect(urlHasCapturedV4ViewState(params('tag=env%3Dprod'))).toBe(true);
  });

  test('false for a bare share key with no view state (a garbage/stale link)', () => {
    expect(urlHasCapturedV4ViewState(params(`${TRACE_V4_SHARE_URL_PARAM_KEY}=x`))).toBe(false);
    expect(urlHasCapturedV4ViewState(params(''))).toBe(false);
    // Transient params alone are not view state.
    expect(urlHasCapturedV4ViewState(params('page=2&traceId=abc'))).toBe(false);
  });
});

describe('decodeViewColumns', () => {
  const withCols = (cols: string) => ({ single: { cols }, multi: {} });

  test('resolves known column ids in the given (stored) order', () => {
    expect(decodeViewColumns(withCols('start_time,input,duration'), TRACE_COLUMN_IDS)).toEqual([
      'start_time',
      'input',
      'duration',
    ]);
  });

  test('drops ids that no longer resolve (view saved against an older column set)', () => {
    expect(decodeViewColumns(withCols('start_time,bogus,input'), TRACE_COLUMN_IDS)).toEqual(['start_time', 'input']);
  });

  test('returns undefined when nothing resolves or the value is empty/absent', () => {
    expect(decodeViewColumns(withCols('bogus'), TRACE_COLUMN_IDS)).toBeUndefined();
    expect(decodeViewColumns(withCols(''), TRACE_COLUMN_IDS)).toBeUndefined();
    expect(decodeViewColumns({ single: {}, multi: {} }, TRACE_COLUMN_IDS)).toBeUndefined();
  });
});

describe('getTraceV4SavedViewShareUrl', () => {
  test('builds an absolute hash-route link carrying the view query and share key', () => {
    const state = captureV4ViewState(params('q=refund&sort=duration'), ['start_time', 'input']);
    const url = getTraceV4SavedViewShareUrl('exp-42', state, 'view-9');
    // Points at the experiment's Traces tab, carries the captured query and the share key.
    expect(url).toContain('exp-42');
    expect(url).toContain('q=refund');
    expect(url).toContain(`${TRACE_V4_SHARE_URL_PARAM_KEY}=view-9`);
  });
});
