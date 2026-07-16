import { jest, describe, it, expect } from '@jest/globals';
import { renderHook, act } from '@testing-library/react';
import { decodePreviewColumns, decodePreviewSort, useSavedViewPreview } from './traceSavedViewPreview';
import { type TracesTableColumn, TracesTableColumnType } from '@databricks/web-shared/genai-traces-table';

const allColumns: TracesTableColumn[] = [
  { id: 'trace_id', label: 'Trace ID', type: TracesTableColumnType.TRACE_INFO },
  { id: 'request_time', label: 'Request time', type: TracesTableColumnType.TRACE_INFO },
  { id: 'execution_duration', label: 'Duration', type: TracesTableColumnType.TRACE_INFO },
];

describe('decodePreviewColumns', () => {
  it('maps a comma-joined id list to the matching column objects, preserving order', () => {
    const result = decodePreviewColumns('execution_duration,trace_id', allColumns);
    expect(result?.map((c) => c.id)).toEqual(['execution_duration', 'trace_id']);
  });

  it('drops ids that do not resolve to a known column', () => {
    const result = decodePreviewColumns('trace_id,ghost_column', allColumns);
    expect(result?.map((c) => c.id)).toEqual(['trace_id']);
  });

  it('returns undefined for a missing or empty value', () => {
    expect(decodePreviewColumns(undefined, allColumns)).toBeUndefined();
    expect(decodePreviewColumns('', allColumns)).toBeUndefined();
  });

  it('returns undefined (not an empty array) when NO id resolves, so the caller falls back', () => {
    // A view saved against an older column schema, or opened before allColumns finished loading.
    expect(decodePreviewColumns('ghost_a,ghost_b', allColumns)).toBeUndefined();
    expect(decodePreviewColumns('trace_id', [])).toBeUndefined();
  });
});

describe('decodePreviewSort', () => {
  it('parses the key::type::asc wire format', () => {
    expect(decodePreviewSort('request_time::TRACE_INFO::false')).toEqual({
      key: 'request_time',
      type: 'TRACE_INFO',
      asc: false,
    });
  });

  it('treats "true" as ascending', () => {
    expect(decodePreviewSort('trace_id::TRACE_INFO::true')?.asc).toBe(true);
  });

  it('returns undefined for a missing value or a malformed triple', () => {
    expect(decodePreviewSort(undefined)).toBeUndefined();
    expect(decodePreviewSort('')).toBeUndefined();
    expect(decodePreviewSort('only_key')).toBeUndefined();
    expect(decodePreviewSort('key::type')).toBeUndefined();
  });
});

describe('useSavedViewPreview', () => {
  const baseArgs = () => ({
    active: true,
    rawColumns: 'execution_duration,trace_id',
    rawSort: 'request_time::TRACE_INFO::false',
    allColumns,
    setSelectedColumns: jest.fn<(columns: TracesTableColumn[]) => void>(),
    setTableSort: jest.fn(),
    exitPreview: jest.fn(),
  });

  it('exposes decoded columns/sort only while active', () => {
    const { result, rerender } = renderHook((args) => useSavedViewPreview(args), { initialProps: baseArgs() });
    expect(result.current.active).toBe(true);
    expect(result.current.columns?.map((c) => c.id)).toEqual(['execution_duration', 'trace_id']);
    expect(result.current.sort).toEqual({ key: 'request_time', type: 'TRACE_INFO', asc: false });

    rerender({ ...baseArgs(), active: false });
    expect(result.current.columns).toBeUndefined();
    expect(result.current.sort).toBeUndefined();
  });

  it('override adopts the preview into the real setters (the only persistence write) then exits', () => {
    const args = baseArgs();
    const { result } = renderHook(() => useSavedViewPreview(args));
    act(() => result.current.override());

    expect(args.setSelectedColumns).toHaveBeenCalledTimes(1);
    expect(args.setSelectedColumns.mock.calls[0][0].map((c) => c.id)).toEqual(['execution_duration', 'trace_id']);
    expect(args.setTableSort).toHaveBeenCalledWith({ key: 'request_time', type: 'TRACE_INFO', asc: false });
    expect(args.exitPreview).toHaveBeenCalledTimes(1);
  });

  it('discard exits preview without writing to the real setters', () => {
    const args = baseArgs();
    const { result } = renderHook(() => useSavedViewPreview(args));
    act(() => result.current.discard());

    expect(args.setSelectedColumns).not.toHaveBeenCalled();
    expect(args.setTableSort).not.toHaveBeenCalled();
    expect(args.exitPreview).toHaveBeenCalledTimes(1);
  });
});
