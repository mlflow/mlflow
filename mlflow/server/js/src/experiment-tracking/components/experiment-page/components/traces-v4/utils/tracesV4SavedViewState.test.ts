import { describe, expect, test } from '@jest/globals';
import { TRACE_COLUMN_IDS } from '@databricks/web-shared/traces-table';
import {
  captureV4ViewState,
  buildV4ViewQuery,
  decodePreviewColumns,
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
        'q=refund&sort=duration&dir=asc&pageSize=50&page=3&traceId=abc123&selectedTraceId=def456&selectedEvaluationId=ghi789&startTimeLabel=LAST_7_DAYS',
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
    // Transient state is never part of a saved view.
    expect(out.get('page')).toBeNull();
    expect(out.get('traceId')).toBeNull();
    expect(out.get('selectedTraceId')).toBeNull();
    expect(out.get('selectedEvaluationId')).toBeNull();
  });

  test('preserves repeatable tag filters in order', () => {
    const state = captureV4ViewState(params('tag=env%3Dprod&tag=team%3Dsearch'), ['start_time']);
    const out = params(buildV4ViewQuery(state, 'view-1'));
    expect(out.getAll('tag')).toEqual(['env=prod', 'team=search']);
  });

  test('captures the live visible columns as the cols param (not read from the URL)', () => {
    // Columns live in localStorage, not the URL, so capture takes them from the live arg.
    const state = captureV4ViewState(params('q=x'), ['start_time', 'input', 'duration']);
    const out = params(buildV4ViewQuery(state, 'view-1'));
    expect(out.get('cols')).toBe('start_time,input,duration');
  });

  test('never captures an incoming share key or cols param from the URL (only the live columns)', () => {
    // Opening view A then saving must not leak A's share key / cols into view B.
    const state = captureV4ViewState(params(`q=x&cols=tokens,cost&${TRACE_V4_SHARE_URL_PARAM_KEY}=other-view`), [
      'start_time',
      'input',
    ]);
    const out = params(buildV4ViewQuery(state, 'view-b'));
    expect(out.get('cols')).toBe('start_time,input'); // live columns win, not the URL's cols
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
    expect(urlHasCapturedV4ViewState(params('cols=start_time'))).toBe(true);
    expect(urlHasCapturedV4ViewState(params('tag=env%3Dprod'))).toBe(true);
  });

  test('false for a bare share key with no view state (a garbage/stale link)', () => {
    expect(urlHasCapturedV4ViewState(params(`${TRACE_V4_SHARE_URL_PARAM_KEY}=x`))).toBe(false);
    expect(urlHasCapturedV4ViewState(params(''))).toBe(false);
    // Transient params alone are not view state.
    expect(urlHasCapturedV4ViewState(params('page=2&traceId=abc&selectedTraceId=def&selectedEvaluationId=ghi'))).toBe(
      false,
    );
  });
});

describe('decodePreviewColumns', () => {
  test('resolves known column ids in the given order', () => {
    expect(decodePreviewColumns('start_time,input,duration', TRACE_COLUMN_IDS)).toEqual([
      'start_time',
      'input',
      'duration',
    ]);
  });

  test('drops ids that no longer resolve (view saved against an older column set)', () => {
    expect(decodePreviewColumns('start_time,bogus,input', TRACE_COLUMN_IDS)).toEqual(['start_time', 'input']);
  });

  test('returns undefined when nothing resolves or the value is empty', () => {
    expect(decodePreviewColumns('bogus', TRACE_COLUMN_IDS)).toBeUndefined();
    expect(decodePreviewColumns('', TRACE_COLUMN_IDS)).toBeUndefined();
    expect(decodePreviewColumns(undefined, TRACE_COLUMN_IDS)).toBeUndefined();
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
