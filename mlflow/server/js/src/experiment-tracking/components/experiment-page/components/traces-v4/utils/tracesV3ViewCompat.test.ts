import { describe, expect, test } from '@jest/globals';
import { FilterOp } from '@databricks/web-shared/traces-table';
import {
  getTraceV3SavedViewIdFromTagKey,
  getTraceV3SavedViewTagKey,
  translateV3ViewState,
  TRACE_V3_SAVED_VIEW_TAG_PREFIX,
} from './tracesV3ViewCompat';
import { buildV4ViewQuery } from './tracesV4SavedViewState';

describe('getTraceV3SavedViewTagKey', () => {
  test('builds a legacy V3 tag key from an id, round-tripping with the id extractor', () => {
    const key = getTraceV3SavedViewTagKey('abc');
    expect(key).toBe(`${TRACE_V3_SAVED_VIEW_TAG_PREFIX}abc`);
    expect(getTraceV3SavedViewIdFromTagKey(key)).toBe('abc');
  });
});

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
  test('maps V3 column ids onto their V4 counterparts, preserving order', () => {
    // V3 named these columns differently: request→input, response→output, request_time→start_time,
    // execution_duration→duration; the rest share the same id.
    const out = translateV3ViewState({
      single: { selectedColumns: 'request_time,request,response,execution_duration,state' },
    });
    expect(out.single.cols).toBe('start_time,input,output,duration,state');
  });

  test('drops V3 column ids with no V4 column (e.g. logged_model, span.*)', () => {
    const out = translateV3ViewState({ single: { selectedColumns: 'request,logged_model,span.name,tokens' } });
    expect(out.single.cols).toBe('input,tokens');
  });

  test('omits cols when nothing maps or V3 had no selectedColumns', () => {
    expect(translateV3ViewState({ single: { selectedColumns: 'logged_model,prompt' } }).single.cols).toBeUndefined();
    expect(translateV3ViewState({ single: { selectedColumns: '' } }).single.cols).toBeUndefined();
    expect(translateV3ViewState({ single: {} }).single.cols).toBeUndefined();
  });
});

describe('translateV3ViewState — sort', () => {
  test('maps a V3 sort key to its V4 column and splits key::type::asc into sort + dir', () => {
    // execution_duration → duration; the middle `type` segment is dropped.
    const out = translateV3ViewState({ single: { sort: 'execution_duration::number::true' } });
    expect(out.single.sort).toBe('duration');
    expect(out.single.dir).toBe('asc');
  });

  test('maps request_time → start_time and asc=false to descending', () => {
    const out = translateV3ViewState({ single: { sort: 'request_time::date::false' } });
    expect(out.single.sort).toBe('start_time');
    expect(out.single.dir).toBe('desc');
  });

  test('drops a sort whose V3 key has no V4-sortable column', () => {
    // `session` is client-sortable in V3 but not a V4-sortable column, so the sort is dropped.
    const out = translateV3ViewState({ single: { sort: 'session::string::true' } });
    expect(out.single.sort).toBeUndefined();
    expect(out.single.dir).toBeUndefined();
  });

  test('drops a malformed sort (wrong segment count) so the view still opens unsorted', () => {
    const out = translateV3ViewState({ single: { sort: 'execution_duration::number' } });
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
  test('translates V3 column::operator::value::key entries into the V4 popover model, in order', () => {
    // V3 serializes each filter as `column::operator::value::key` (the key segment is empty for
    // non-key fields). state→state, execution_duration→duration.
    const out = translateV3ViewState({
      multi: { filter: ['state::=::ERROR::', 'execution_duration::>=::1000::'] },
    });
    expect(out.filters).toEqual([
      { field: 'state', operator: FilterOp.EQUALS, value: 'ERROR' },
      { field: 'duration', operator: FilterOp.GREATER_THAN_OR_EQUALS, value: '1000' },
    ]);
  });

  test('carries the key segment for a key-requiring field (TAG) and maps the column group', () => {
    const out = translateV3ViewState({ multi: { filter: ['TAG::=::prod::env'] } });
    expect(out.filters).toEqual([{ field: 'tag', operator: FilterOp.EQUALS, value: 'prod', key: 'env' }]);
  });

  test('maps the ASSESSMENT group to the assessment field, keeping its key', () => {
    const out = translateV3ViewState({ multi: { filter: ['ASSESSMENT::=::yes::correctness'] } });
    expect(out.filters).toEqual([{ field: 'assessment', operator: FilterOp.EQUALS, value: 'yes', key: 'correctness' }]);
  });

  test('drops entries whose column has no V4 field, whose operator V4 rejects, or that lack a value/key', () => {
    const out = translateV3ViewState({
      multi: {
        filter: [
          'logged_model::=::m1::', // column has no V4 filter field
          'state::IS NULL::::', // operator V4 doesn't support
          'user::=::::', // missing value
          'TAG::=::prod::', // key-requiring field with no key
          'session::=::sess-1::', // valid — kept
        ],
      },
    });
    expect(out.filters).toEqual([{ field: 'session', operator: FilterOp.EQUALS, value: 'sess-1' }]);
  });

  test('omits filters entirely when nothing maps, and never sets the URL-backed tag[]', () => {
    const out = translateV3ViewState({ multi: { filter: ['logged_model::=::m1::'] } });
    expect(out.filters).toBeUndefined();
    expect(out.multi.tag).toBeUndefined();
  });

  test('omits filters when V3 had no filters', () => {
    expect(translateV3ViewState({ single: {} }).filters).toBeUndefined();
  });

  test('drops V3-internal fields with no V4 equivalent (viewState)', () => {
    const out = translateV3ViewState({ single: { viewState: 'whatever' } as any });
    expect((out.single as Record<string, unknown>)['viewState']).toBeUndefined();
  });
});

describe('translateV3ViewState — end to end', () => {
  test('a full V3 view translates into a valid, applyable V4 query + popover filter model', () => {
    const v3State = {
      single: {
        selectedColumns: 'request_time,request,execution_duration',
        sort: 'execution_duration::number::false',
        viewState: 'internal-v3-only',
        startTimeLabel: 'LAST_7_DAYS',
      },
      multi: { filter: ['state::=::ERROR::', 'TAG::=::prod::env'] },
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
    // V3 filters translate into V4's in-memory popover model, NOT the URL-backed tag[].
    expect(out.filters).toEqual([
      { field: 'state', operator: FilterOp.EQUALS, value: 'ERROR' },
      { field: 'tag', operator: FilterOp.EQUALS, value: 'prod', key: 'env' },
    ]);
    expect(params.getAll('tag')).toEqual([]);
    // The share key is set so the applied view is recognized as a preview.
    expect(params.get('traceViewShareKey')).toBe('legacy-view-1');
    // V3-internal state never leaks into the V4 URL.
    expect(params.get('viewState')).toBeNull();
  });

  test('an empty V3 view yields an empty captured state (no crash)', () => {
    const out = translateV3ViewState({});
    expect(out.single).toEqual({});
    expect(out.multi).toEqual({});
    expect(out.filters).toBeUndefined();
  });
});
