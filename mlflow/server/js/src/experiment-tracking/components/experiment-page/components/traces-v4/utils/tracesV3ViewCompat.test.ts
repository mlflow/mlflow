import { describe, expect, test } from '@jest/globals';
import {
  getTraceV3SavedViewIdFromTagKey,
  translateV3ViewState,
  TRACE_V3_SAVED_VIEW_TAG_PREFIX,
} from './tracesV3ViewCompat';
import { buildV4ViewQuery } from './tracesV4SavedViewState';

describe('getTraceV3SavedViewIdFromTagKey', () => {
  test('extracts the id from a legacy V3 tag key', () => {
    expect(getTraceV3SavedViewIdFromTagKey(`${TRACE_V3_SAVED_VIEW_TAG_PREFIX}abc`)).toBe('abc');
  });

  test('a key that is exactly the prefix (no id) is not a saved-view key', () => {
    expect(getTraceV3SavedViewIdFromTagKey(TRACE_V3_SAVED_VIEW_TAG_PREFIX)).toBeNull();
  });

  test('rejects V4 and runs prefixes so the V3 reader only claims V3 tags', () => {
    expect(getTraceV3SavedViewIdFromTagKey('mlflow.tracesV4ViewState.v1')).toBeNull();
    expect(getTraceV3SavedViewIdFromTagKey('mlflow.sharedViewState.v1')).toBeNull();
    expect(getTraceV3SavedViewIdFromTagKey('mlflow.note')).toBeNull();
  });
});

describe('translateV3ViewState — columns', () => {
  test('maps V3 selectedColumns onto V4 cols verbatim', () => {
    const out = translateV3ViewState({ single: { selectedColumns: 'start_time,input,duration' } });
    expect(out.single.cols).toBe('start_time,input,duration');
  });

  test('preserves an empty selectedColumns string (every column deselected) rather than dropping it', () => {
    const out = translateV3ViewState({ single: { selectedColumns: '' } });
    expect(out.single.cols).toBe('');
  });

  test('omits cols when V3 had no selectedColumns', () => {
    const out = translateV3ViewState({ single: {} });
    expect(out.single.cols).toBeUndefined();
  });
});

describe('translateV3ViewState — sort', () => {
  test('splits an ascending key::type::asc string into sort + dir, dropping the type segment', () => {
    const out = translateV3ViewState({ single: { sort: 'duration::numeric::true' } });
    expect(out.single.sort).toBe('duration');
    expect(out.single.dir).toBe('asc');
  });

  test('maps asc=false to descending', () => {
    const out = translateV3ViewState({ single: { sort: 'timestamp::date::false' } });
    expect(out.single.sort).toBe('timestamp');
    expect(out.single.dir).toBe('desc');
  });

  test('drops a malformed sort (wrong segment count) so the view still opens unsorted', () => {
    const out = translateV3ViewState({ single: { sort: 'duration::numeric' } });
    expect(out.single.sort).toBeUndefined();
    expect(out.single.dir).toBeUndefined();
  });

  test('drops a sort with an empty key', () => {
    const out = translateV3ViewState({ single: { sort: '::numeric::true' } });
    expect(out.single.sort).toBeUndefined();
    expect(out.single.dir).toBeUndefined();
  });

  test('omits sort/dir when V3 had no sort', () => {
    const out = translateV3ViewState({ single: {} });
    expect(out.single.sort).toBeUndefined();
    expect(out.single.dir).toBeUndefined();
  });
});

describe('translateV3ViewState — time range', () => {
  test('passes the time-range fields through unchanged (identical keys/semantics)', () => {
    const out = translateV3ViewState({
      single: { startTimeLabel: 'CUSTOM', startTime: '2024-01-01T00:00:00Z', endTime: '2024-01-08T00:00:00Z' },
    });
    expect(out.single.startTimeLabel).toBe('CUSTOM');
    expect(out.single.startTime).toBe('2024-01-01T00:00:00Z');
    expect(out.single.endTime).toBe('2024-01-08T00:00:00Z');
  });
});

describe('translateV3ViewState — filters', () => {
  test('maps key=value V3 filters onto V4 tags, preserving order', () => {
    const out = translateV3ViewState({ multi: { filter: ['env=prod', 'team=search'] } });
    expect(out.multi.tag).toEqual(['env=prod', 'team=search']);
  });

  test('drops filter entries that are not key=value', () => {
    const out = translateV3ViewState({
      multi: { filter: ['env=prod', 'status:error', 'plainword', '=novalue', 'nokey='] },
    });
    expect(out.multi.tag).toEqual(['env=prod']);
  });

  test('omits tag entirely when no filter maps', () => {
    const out = translateV3ViewState({ multi: { filter: ['plainword'] } });
    expect(out.multi.tag).toBeUndefined();
  });

  test('omits tag when V3 had no filters', () => {
    const out = translateV3ViewState({ single: {} });
    expect(out.multi.tag).toBeUndefined();
  });

  test('drops V3-internal fields with no V4 equivalent (viewState)', () => {
    const out = translateV3ViewState({ single: { viewState: 'whatever' } as any });
    expect((out.single as Record<string, unknown>)['viewState']).toBeUndefined();
  });
});

describe('translateV3ViewState — end to end', () => {
  test('a full V3 view translates into a valid, applyable V4 query', () => {
    const v3State = {
      single: {
        selectedColumns: 'start_time,input,duration',
        sort: 'duration::numeric::false',
        viewState: 'internal-v3-only',
        startTimeLabel: 'LAST_7_DAYS',
      },
      multi: { filter: ['env=prod'] },
    };

    const out = translateV3ViewState(v3State);
    const params = new URLSearchParams(buildV4ViewQuery(out, 'legacy-view-1'));

    // Columns are stored on the envelope (restored into the column store on open), not carried in
    // the URL query — so they live in `single.cols`, not as a `cols` param.
    expect(out.single.cols).toBe('start_time,input,duration');
    expect(params.get('cols')).toBeNull();
    expect(params.get('sort')).toBe('duration');
    expect(params.get('dir')).toBe('desc');
    expect(params.get('startTimeLabel')).toBe('LAST_7_DAYS');
    expect(params.getAll('tag')).toEqual(['env=prod']);
    // V3's key=value filters map to V4's URL-backed tag[], NOT the in-memory popover filter model
    // (which V3 had no equivalent of) — so a translated view carries no popover `filters`.
    expect(out.filters).toBeUndefined();
    // The share key is set so the applied view is recognized as a preview.
    expect(params.get('traceViewShareKey')).toBe('legacy-view-1');
    // V3-internal state never leaks into the V4 URL.
    expect(params.get('viewState')).toBeNull();
  });

  test('an empty V3 view yields an empty captured state (no crash)', () => {
    const out = translateV3ViewState({});
    expect(out.single).toEqual({});
    expect(out.multi).toEqual({});
  });
});
