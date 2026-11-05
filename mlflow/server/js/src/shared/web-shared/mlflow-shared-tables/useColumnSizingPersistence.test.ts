import { describe, expect, it, jest } from '@jest/globals';
import type { ColumnSizingState, Table } from '@tanstack/react-table';
import { renderHook } from '@testing-library/react';

import { useColumnSizingPersistence } from './useColumnSizingPersistence';

const createTable = (columnSizing: ColumnSizingState, clampedSizes: Record<string, number>) => {
  const state = { columnSizing, columnSizingInfo: { isResizingColumn: false as string | false } };
  const table = {
    getState: () => state,
    getAllLeafColumns: () => Object.entries(clampedSizes).map(([id, size]) => ({ id, getSize: () => size })),
    setColumnSizing: jest.fn(),
  };
  return { table, state };
};

describe('useColumnSizingPersistence', () => {
  it('does not persist on mount', () => {
    const { table } = createTable({}, { name: 150 });
    const onSettled = jest.fn();

    renderHook(() => useColumnSizingPersistence(table as unknown as Table<unknown>, onSettled));

    expect(onSettled).not.toHaveBeenCalled();
  });

  it('persists the clamped sizes once when a resize ends', () => {
    const { table, state } = createTable({ name: 900 }, { name: 320, other: 100 });
    const onSettled = jest.fn();
    const { rerender } = renderHook(() => useColumnSizingPersistence(table as unknown as Table<unknown>, onSettled));

    state.columnSizingInfo.isResizingColumn = 'name';
    rerender();
    expect(onSettled).not.toHaveBeenCalled();

    state.columnSizingInfo.isResizingColumn = false;
    rerender();
    rerender();

    expect(onSettled).toHaveBeenCalledTimes(1);
    expect(onSettled).toHaveBeenCalledWith({ name: 320 });
    expect(table.setColumnSizing).toHaveBeenCalledWith({ name: 320 });
  });
});
