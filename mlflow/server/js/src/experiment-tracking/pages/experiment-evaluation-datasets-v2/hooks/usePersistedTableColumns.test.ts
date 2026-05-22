/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck — punting test typing; see PR2 plan in branch import { afterEach, beforeEach, describe, expect, test } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';
import { usePersistedTableColumns } from './usePersistedTableColumns';
import { describe } from '@jest/globals';
import { beforeEach } from '@jest/globals';
import { afterEach } from '@jest/globals';
import { test } from '@jest/globals';
import { expect } from '@jest/globals';

const COLUMNS = ['inputs', 'expectations', 'last_updated', 'source', 'tags'] as const;
const DEFAULTS = ['inputs', 'expectations', 'last_updated', 'source', 'tags'] as const;

describe('usePersistedTableColumns', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });
  afterEach(() => {
    window.localStorage.clear();
  });

  test('returns defaults when nothing is persisted', () => {
    const { result } = renderHook(() =>
      usePersistedTableColumns({
        experimentId: 'exp-1',
        datasetId: 'ds-1',
        allColumns: COLUMNS,
        defaultVisible: DEFAULTS,
      }),
    );
    expect(result.current.visibleColumns).toEqual(DEFAULTS);
  });

  test('toggling a column removes it; toggling again restores it in canonical order', () => {
    const { result } = renderHook(() =>
      usePersistedTableColumns({
        experimentId: 'exp-1',
        datasetId: 'ds-1',
        allColumns: COLUMNS,
        defaultVisible: DEFAULTS,
      }),
    );

    act(() => {
      result.current.toggleColumn('source');
    });
    expect(result.current.visibleColumns).toEqual(['inputs', 'expectations', 'last_updated', 'tags']);

    act(() => {
      result.current.toggleColumn('source');
    });
    // Restored in canonical (allColumns) order, not appended at the end.
    expect(result.current.visibleColumns).toEqual(['inputs', 'expectations', 'last_updated', 'source', 'tags']);
  });

  test('persists across hook remounts for the same dataset', () => {
    const params = {
      experimentId: 'exp-1',
      datasetId: 'ds-1',
      allColumns: COLUMNS,
      defaultVisible: DEFAULTS,
    } as const;

    const first = renderHook(() => usePersistedTableColumns(params));
    act(() => {
      first.result.current.toggleColumn('tags');
    });
    first.unmount();

    const second = renderHook(() => usePersistedTableColumns(params));
    expect(second.result.current.visibleColumns).toEqual(['inputs', 'expectations', 'last_updated', 'source']);
  });

  test('keys are isolated per dataset', () => {
    const firstParams = {
      experimentId: 'exp-1',
      datasetId: 'ds-1',
      allColumns: COLUMNS,
      defaultVisible: DEFAULTS,
    } as const;
    const first = renderHook(() => usePersistedTableColumns(firstParams));
    act(() => {
      first.result.current.toggleColumn('source');
    });

    // Different dataset → starts from defaults.
    const secondParams = {
      experimentId: 'exp-1',
      datasetId: 'ds-2',
      allColumns: COLUMNS,
      defaultVisible: DEFAULTS,
    } as const;
    const second = renderHook(() => usePersistedTableColumns(secondParams));
    expect(second.result.current.visibleColumns).toEqual(DEFAULTS);
  });

  test('resetToDefaults clears customization', () => {
    const params = {
      experimentId: 'exp-1',
      datasetId: 'ds-1',
      allColumns: COLUMNS,
      defaultVisible: DEFAULTS,
    } as const;
    const { result } = renderHook(() => usePersistedTableColumns(params));

    act(() => {
      result.current.toggleColumn('inputs');
      result.current.toggleColumn('source');
    });
    expect(result.current.visibleColumns).not.toEqual(DEFAULTS);

    act(() => {
      result.current.resetToDefaults();
    });
    expect(result.current.visibleColumns).toEqual(DEFAULTS);
  });

  test('isVisible reflects current state', () => {
    const { result } = renderHook(() =>
      usePersistedTableColumns({
        experimentId: 'exp-1',
        datasetId: 'ds-1',
        allColumns: COLUMNS,
        defaultVisible: ['inputs', 'expectations'],
      }),
    );
    expect(result.current.isVisible('inputs')).toBe(true);
    expect(result.current.isVisible('source')).toBe(false);
  });

  test('adding a new column to allColumns preserves existing persisted visibility', () => {
    // Lock in the no-bump invariant: when a future change extends the column set, users
    // with an existing customized visibility should keep their selection and the new
    // column should simply be off by default until they toggle it on.
    const initialParams = {
      experimentId: 'exp-1',
      datasetId: 'ds-1',
      allColumns: COLUMNS,
      defaultVisible: DEFAULTS,
    } as const;
    const first = renderHook(() => usePersistedTableColumns(initialParams));
    act(() => {
      first.result.current.toggleColumn('source');
    });
    expect(first.result.current.visibleColumns).toEqual(['inputs', 'expectations', 'last_updated', 'tags']);
    first.unmount();

    const extendedColumns = ['inputs', 'expectations', 'last_updated', 'created_by', 'source', 'tags'] as const;
    const second = renderHook(() =>
      usePersistedTableColumns({
        experimentId: 'exp-1',
        datasetId: 'ds-1',
        allColumns: extendedColumns,
        defaultVisible: DEFAULTS,
      }),
    );
    expect(second.result.current.visibleColumns).toEqual(['inputs', 'expectations', 'last_updated', 'tags']);
    expect(second.result.current.isVisible('created_by')).toBe(false);

    // Toggling the new column on inserts it in canonical position, not at the end.
    act(() => {
      second.result.current.toggleColumn('created_by');
    });
    expect(second.result.current.visibleColumns).toEqual([
      'inputs',
      'expectations',
      'last_updated',
      'created_by',
      'tags',
    ]);
  });
});
