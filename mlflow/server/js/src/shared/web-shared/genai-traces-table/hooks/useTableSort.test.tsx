import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';

import { useTableSort } from './useTableSort';
import type { EvaluationsOverviewTableSort, TracesTableColumn } from '../types';
import { TracesTableColumnType } from '../types';

jest.mock('../../model-trace-explorer/FeatureUtils', () => ({
  shoudlEnableURLPersistenceForSortAndColumns: () => true,
}));

const mockColumns: TracesTableColumn[] = [
  { id: 'col1', type: TracesTableColumnType.ASSESSMENT, label: 'Assessment 1' },
  { id: 'col2', type: TracesTableColumnType.ASSESSMENT, label: 'Assessment 2' },
  { id: 'col3', type: TracesTableColumnType.TRACE_INFO, label: 'Info 1' },
  { id: 'col4', type: TracesTableColumnType.INPUT, label: 'Input 1' },
];

const createSort = (columnId: string, asc = true): EvaluationsOverviewTableSort => ({
  key: columnId,
  type: TracesTableColumnType.ASSESSMENT,
  asc,
});

let mockUrlSort: EvaluationsOverviewTableSort | undefined;
let mockSetUrlSort: jest.MockedFunction<(sort: EvaluationsOverviewTableSort | undefined, replace?: boolean) => void>;

jest.mock('./useTableSortURL', () => ({
  useTableSortURL: () => {
    mockSetUrlSort = jest.fn((sort) => {
      mockUrlSort = sort;
    });
    return [mockUrlSort, mockSetUrlSort] as const;
  },
}));

describe('useTableSort', () => {
  beforeEach(() => {
    mockUrlSort = undefined;
    mockSetUrlSort = jest.fn();
  });

  describe('Priority: URL → initial → undefined', () => {
    it('returns undefined when no URL and no initial sort', () => {
      const { result } = renderHook(() => useTableSort(mockColumns, undefined));

      expect(result.current[0]).toBeUndefined();
    });

    it('uses initial sort when URL is empty', () => {
      const initialSort = createSort('col1');
      const { result } = renderHook(() => useTableSort(mockColumns, initialSort));

      expect(result.current[0]).toEqual(initialSort);
    });

    it('prefers URL sort over initial sort', () => {
      const initialSort = createSort('col1');
      const urlSort = createSort('col2', false);
      mockUrlSort = urlSort;

      const { result } = renderHook(() => useTableSort(mockColumns, initialSort));

      expect(result.current[0]).toEqual(urlSort);
    });

    it('falls back to initial when URL sort column not visible', () => {
      const initialSort = createSort('col1');
      mockUrlSort = createSort('col_invalid');

      const { result } = renderHook(() => useTableSort(mockColumns, initialSort));

      expect(result.current[0]).toEqual(initialSort);
    });
  });

  describe('Validation: sort column must be visible', () => {
    it('clears sort when sorted column is deselected', () => {
      const initialSort = createSort('col1');
      const { result, rerender } = renderHook(({ columns }) => useTableSort(columns, initialSort), {
        initialProps: { columns: mockColumns },
      });

      expect(result.current[0]).toEqual(initialSort);

      const columnsWithoutCol1 = mockColumns.filter((c) => c.id !== 'col1');
      rerender({ columns: columnsWithoutCol1 });

      expect(result.current[0]).toBeUndefined();
    });

    it('reacts to URL sort changes when columns change', () => {
      mockUrlSort = createSort('col3');
      const { result, rerender } = renderHook(({ columns }) => useTableSort(columns, undefined), {
        initialProps: { columns: mockColumns },
      });

      expect(result.current[0]?.key).toBe('col3');

      const columnsWithoutCol3 = mockColumns.filter((c) => c.id !== 'col3');
      rerender({ columns: columnsWithoutCol3 });

      expect(result.current[0]).toBeUndefined();
    });
  });

  describe('URL updates via setTableSort', () => {
    it('writes new sort to URL', () => {
      const { result } = renderHook(() => useTableSort(mockColumns, undefined));
      const newSort = createSort('col2', false);

      act(() => {
        result.current[1](newSort);
      });

      expect(mockSetUrlSort).toHaveBeenCalledWith(newSort, false);
    });

    it('clears URL when sort is removed', () => {
      const { result } = renderHook(() => useTableSort(mockColumns, createSort('col1')));

      act(() => {
        result.current[1](undefined);
      });

      expect(mockSetUrlSort).toHaveBeenCalledWith(undefined, false);
    });
  });
});
