import { afterEach, beforeEach, describe, expect, test } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';
import { usePersistedTableColumnWidths } from './usePersistedTableColumnWidths';

describe('usePersistedTableColumnWidths', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });
  afterEach(() => {
    window.localStorage.clear();
  });

  test('starts empty when nothing is persisted', () => {
    // Defaults live on the ColumnDefs, so the stored map begins empty and only dragged
    // columns ever get written.
    const { result } = renderHook(() => usePersistedTableColumnWidths({ experimentId: 'exp-1', datasetId: 'ds-1' }));
    expect(result.current.columnSizing).toEqual({});
  });

  test('setColumnSizing stores a direct value', () => {
    const { result } = renderHook(() => usePersistedTableColumnWidths({ experimentId: 'exp-1', datasetId: 'ds-1' }));

    act(() => {
      result.current.setColumnSizing({ inputs: 512 });
    });

    expect(result.current.columnSizing).toEqual({ inputs: 512 });
  });

  test('setColumnSizing accepts a functional updater (the shape TanStack calls it with)', () => {
    // onColumnSizingChange hands the setter a `(prev) => next` updater on every drag tick;
    // this guards that the useLocalStorage setter plugs in directly with no adapter.
    const { result } = renderHook(() => usePersistedTableColumnWidths({ experimentId: 'exp-1', datasetId: 'ds-1' }));

    act(() => {
      result.current.setColumnSizing({ inputs: 512 });
    });
    act(() => {
      result.current.setColumnSizing((prev) => ({ ...prev, expectations: 320 }));
    });

    expect(result.current.columnSizing).toEqual({ inputs: 512, expectations: 320 });
  });

  test('persists across remounts for the same dataset', () => {
    const params = { experimentId: 'exp-1', datasetId: 'ds-1' };

    const first = renderHook(() => usePersistedTableColumnWidths(params));
    act(() => {
      first.result.current.setColumnSizing({ inputs: 256 });
    });
    first.unmount();

    const second = renderHook(() => usePersistedTableColumnWidths(params));
    expect(second.result.current.columnSizing).toEqual({ inputs: 256 });
  });

  test('widths are isolated per dataset', () => {
    const first = renderHook(() => usePersistedTableColumnWidths({ experimentId: 'exp-1', datasetId: 'ds-1' }));
    act(() => {
      first.result.current.setColumnSizing({ inputs: 256 });
    });

    // Different dataset → its own key → starts empty.
    const second = renderHook(() => usePersistedTableColumnWidths({ experimentId: 'exp-1', datasetId: 'ds-2' }));
    expect(second.result.current.columnSizing).toEqual({});
  });

  test('widths are isolated per experiment', () => {
    const first = renderHook(() => usePersistedTableColumnWidths({ experimentId: 'exp-1', datasetId: 'ds-1' }));
    act(() => {
      first.result.current.setColumnSizing({ inputs: 256 });
    });

    const second = renderHook(() => usePersistedTableColumnWidths({ experimentId: 'exp-2', datasetId: 'ds-1' }));
    expect(second.result.current.columnSizing).toEqual({});
  });
});
