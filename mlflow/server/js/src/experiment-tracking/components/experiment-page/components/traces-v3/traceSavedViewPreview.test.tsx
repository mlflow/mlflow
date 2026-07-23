import { jest, describe, it, expect } from '@jest/globals';
import type { ReactNode } from 'react';
import { renderHook, act } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { decodePreviewColumns, decodePreviewSort, useSavedViewPreview } from './traceSavedViewPreview';
import { type TracesTableColumn, TracesTableColumnType } from '@databricks/web-shared/genai-traces-table';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';

const allColumns: TracesTableColumn[] = [
  { id: 'trace_id', label: 'Trace ID', type: TracesTableColumnType.TRACE_INFO },
  { id: 'request_time', label: 'Request time', type: TracesTableColumnType.TRACE_INFO },
  { id: 'execution_duration', label: 'Duration', type: TracesTableColumnType.TRACE_INFO },
];

// useSavedViewPreview reads useDesignSystemTheme (Undo toast) and useIntl (error toast), so it must
// render inside a DesignSystemProvider and an IntlProvider.
const wrapper = ({ children }: { children: ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

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
    // The user's own selection before opening the shared view (only trace_id, sorted by request_time asc).
    ownColumns: [allColumns[0]],
    ownSort: { key: 'request_time', type: TracesTableColumnType.TRACE_INFO, asc: true },
    setSelectedColumns: jest.fn<(columns: TracesTableColumn[]) => void>(),
    setTableSort: jest.fn(),
    exitPreview: jest.fn(),
  });

  it('exposes decoded columns/sort only while active', () => {
    const { result, rerender } = renderHook((args) => useSavedViewPreview(args), {
      initialProps: baseArgs(),
      wrapper,
    });
    expect(result.current.active).toBe(true);
    expect(result.current.columns?.map((c) => c.id)).toEqual(['execution_duration', 'trace_id']);
    expect(result.current.sort).toEqual({ key: 'request_time', type: 'TRACE_INFO', asc: false });

    rerender({ ...baseArgs(), active: false });
    expect(result.current.columns).toBeUndefined();
    expect(result.current.sort).toBeUndefined();
  });

  it('override adopts the preview into the real setters (the only persistence write) then exits', () => {
    const args = baseArgs();
    const { result } = renderHook(() => useSavedViewPreview(args), { wrapper });
    act(() => result.current.override());

    expect(args.setSelectedColumns).toHaveBeenCalledTimes(1);
    expect(args.setSelectedColumns.mock.calls[0][0].map((c) => c.id)).toEqual(['execution_duration', 'trace_id']);
    expect(args.setTableSort).toHaveBeenCalledWith({ key: 'request_time', type: 'TRACE_INFO', asc: false });
    expect(args.exitPreview).toHaveBeenCalledTimes(1);
  });

  it('override shows an Undo toast that restores the user’s pre-override columns/sort', () => {
    const infoSpy = jest.spyOn(Utils, 'displayGlobalInfoNotification').mockImplementation(() => {});
    const args = baseArgs();
    const { result } = renderHook(() => useSavedViewPreview(args), { wrapper });
    act(() => result.current.override());

    // The toast is a React node; find the Undo button in its children and invoke its onClick to
    // exercise the restore closure (rendering the toast itself is the notification system's job).
    expect(infoSpy).toHaveBeenCalledTimes(1);
    const toastNode = infoSpy.mock.calls[0][0] as React.ReactElement;
    const undo = (toastNode.props.children as React.ReactElement[]).find(
      (child) => child?.props?.componentId === 'mlflow.traces.shared_view.override_undo',
    );
    act(() => undo?.props.onClick());

    // Restore writes the user's OWN pre-override selection (trace_id only, request_time asc).
    expect(args.setSelectedColumns).toHaveBeenLastCalledWith(args.ownColumns);
    expect(args.setTableSort).toHaveBeenLastCalledWith(args.ownSort);
    infoSpy.mockRestore();
  });

  it('discard exits preview without writing to the real setters', () => {
    const args = baseArgs();
    const { result } = renderHook(() => useSavedViewPreview(args), { wrapper });
    act(() => result.current.discard());

    expect(args.setSelectedColumns).not.toHaveBeenCalled();
    expect(args.setTableSort).not.toHaveBeenCalled();
    expect(args.exitPreview).toHaveBeenCalledTimes(1);
  });

  it('override with nothing decoded shows an error and stays in preview (no write, no exit)', () => {
    const errorSpy = jest.spyOn(Utils, 'displayGlobalErrorNotification').mockImplementation(() => {});
    // Both raw values unresolvable (e.g. Override clicked before allColumns loaded).
    const args = { ...baseArgs(), rawColumns: 'ghost_a,ghost_b', rawSort: 'malformed' };
    const { result } = renderHook(() => useSavedViewPreview(args), { wrapper });
    act(() => result.current.override());

    expect(errorSpy).toHaveBeenCalledTimes(1);
    expect(args.setSelectedColumns).not.toHaveBeenCalled();
    expect(args.setTableSort).not.toHaveBeenCalled();
    expect(args.exitPreview).not.toHaveBeenCalled();
    errorSpy.mockRestore();
  });
});
