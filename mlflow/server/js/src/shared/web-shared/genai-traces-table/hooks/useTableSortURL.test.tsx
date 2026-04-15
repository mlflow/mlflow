import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';

import { useTableSortURL } from './useTableSortURL';
import type { EvaluationsOverviewTableSort } from '../types';
import { TracesTableColumnType } from '../types';

const createSort = (key: string, type: string, asc: boolean): EvaluationsOverviewTableSort => ({
  key,
  type: type as any,
  asc,
});

let mockSearchParams: URLSearchParams;
let mockSetSearchParams: jest.MockedFunction<
  (updateFn: (params: URLSearchParams) => URLSearchParams, options?: { replace?: boolean }) => void
>;

jest.mock('../utils/RoutingUtils', () => ({
  useSearchParams: () => {
    mockSetSearchParams = jest.fn((updateFn, options) => {
      mockSearchParams = updateFn(mockSearchParams);
    });
    return [mockSearchParams, mockSetSearchParams] as const;
  },
}));

describe('useTableSortURL', () => {
  beforeEach(() => {
    mockSearchParams = new URLSearchParams();
    mockSetSearchParams = jest.fn();
  });

  describe('Reading from URL', () => {
    it('returns undefined when no sort param', () => {
      const { result } = renderHook(() => useTableSortURL());

      expect(result.current[0]).toBeUndefined();
    });

    it('parses format: key::type::asc', () => {
      mockSearchParams.set('sort', 'execution_time::TRACE_INFO::false');
      const { result } = renderHook(() => useTableSortURL());

      expect(result.current[0]).toEqual({
        key: 'execution_time',
        type: 'TRACE_INFO',
        asc: false,
      });
    });

    it('returns undefined for invalid format', () => {
      const invalidFormats = ['col1::ASSESSMENT', '', '::ASSESSMENT::true'];

      invalidFormats.forEach((format) => {
        mockSearchParams = new URLSearchParams();
        mockSearchParams.set('sort', format);
        const { result } = renderHook(() => useTableSortURL());

        expect(result.current[0]).toBeUndefined();
      });
    });
  });

  describe('Writing to URL', () => {
    it('writes sort in format: key::type::asc', () => {
      const { result } = renderHook(() => useTableSortURL());

      act(() => {
        result.current[1](createSort('col2', TracesTableColumnType.ASSESSMENT, false));
      });

      expect(mockSearchParams.get('sort')).toBe('col2::ASSESSMENT::false');
    });

    it('removes param when set to undefined', () => {
      mockSearchParams.set('sort', 'col1::ASSESSMENT::true');
      const { result } = renderHook(() => useTableSortURL());

      act(() => {
        result.current[1](undefined);
      });

      expect(mockSearchParams.has('sort')).toBe(false);
    });

    it('preserves other URL params', () => {
      mockSearchParams.set('filter', 'execution_time::>=::1000');
      mockSearchParams.set('selectedColumns', 'col1,col2,col3');
      const { result } = renderHook(() => useTableSortURL());

      act(() => {
        result.current[1](createSort('col2', 'INPUT', false));
      });

      expect(mockSearchParams.get('filter')).toBe('execution_time::>=::1000');
      expect(mockSearchParams.get('selectedColumns')).toBe('col1,col2,col3');
      expect(mockSearchParams.get('sort')).toBe('col2::INPUT::false');
    });

    it('respects replace option (defaults to false)', () => {
      const { result } = renderHook(() => useTableSortURL());

      act(() => {
        result.current[1](createSort('col1', 'ASSESSMENT', true));
      });
      expect(mockSetSearchParams).toHaveBeenCalledWith(expect.any(Function), { replace: false });

      act(() => {
        result.current[1](createSort('col1', 'ASSESSMENT', true), true);
      });
      expect(mockSetSearchParams).toHaveBeenCalledWith(expect.any(Function), { replace: true });
    });
  });
});
