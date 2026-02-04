import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { renderHook } from '@testing-library/react';

import { useColumnsURL } from './useColumnsURL';

let mockSearchParams: URLSearchParams;
let mockSetSearchParams: jest.MockedFunction<
  (updateFn: (params: URLSearchParams) => URLSearchParams, options?: { replace?: boolean }) => void
>;

jest.mock('../utils/RoutingUtils', () => ({
  useSearchParams: () => {
    mockSetSearchParams = jest.fn((updateFn) => {
      mockSearchParams = updateFn(mockSearchParams);
    });
    return [mockSearchParams, mockSetSearchParams] as const;
  },
}));

describe('useColumnsURL', () => {
  const renderColumnsHook = () => {
    const { result } = renderHook(() => useColumnsURL());
    return {
      columnIds: result.current[0],
      setColumnIds: result.current[1],
    };
  };

  beforeEach(() => {
    mockSearchParams = new URLSearchParams();
    jest.clearAllMocks();
  });

  describe('Reading from URL', () => {
    it('returns undefined when no param', () => {
      const { columnIds } = renderColumnsHook();
      expect(columnIds).toBeUndefined();
    });

    it('parses comma-separated IDs', () => {
      mockSearchParams.set('selectedColumns', 'col1,col2,col3');
      const { columnIds } = renderColumnsHook();
      expect(columnIds).toEqual(['col1', 'col2', 'col3']);
    });

    it('handles single column', () => {
      mockSearchParams.set('selectedColumns', 'single_column');
      const { columnIds } = renderColumnsHook();
      expect(columnIds).toEqual(['single_column']);
    });

    it('filters empty strings from malformed input', () => {
      mockSearchParams.set('selectedColumns', 'col1,,col2,');
      const { columnIds } = renderColumnsHook();
      expect(columnIds).toEqual(['col1', 'col2']);
    });

    it('returns undefined for empty value', () => {
      mockSearchParams.set('selectedColumns', '');
      const { columnIds } = renderColumnsHook();
      expect(columnIds).toBeUndefined();
    });
  });

  describe('Writing to URL', () => {
    it('sets param with comma-separated IDs', () => {
      const { setColumnIds } = renderColumnsHook();

      setColumnIds(['col1', 'col2', 'col3']);

      expect(mockSetSearchParams).toHaveBeenCalledWith(expect.any(Function), { replace: false });
      expect(mockSearchParams.get('selectedColumns')).toBe('col1,col2,col3');
    });

    it('removes param when undefined or empty array', () => {
      mockSearchParams.set('selectedColumns', 'col1,col2');
      const { setColumnIds } = renderColumnsHook();

      setColumnIds(undefined);
      expect(mockSearchParams.has('selectedColumns')).toBe(false);

      mockSearchParams.set('selectedColumns', 'col1,col2');
      setColumnIds([]);
      expect(mockSearchParams.has('selectedColumns')).toBe(false);
    });

    it('respects replace option', () => {
      const { setColumnIds } = renderColumnsHook();

      setColumnIds(['col1'], true);
      expect(mockSetSearchParams).toHaveBeenCalledWith(expect.any(Function), { replace: true });

      setColumnIds(['col2']);
      expect(mockSetSearchParams).toHaveBeenCalledWith(expect.any(Function), { replace: false });
    });
  });

  describe('URL parameter isolation', () => {
    it('updates existing param', () => {
      mockSearchParams.set('selectedColumns', 'col1,col2');
      const { setColumnIds } = renderColumnsHook();

      setColumnIds(['col3', 'col4']);

      expect(mockSearchParams.get('selectedColumns')).toBe('col3,col4');
    });

    it('preserves other params when setting', () => {
      mockSearchParams.set('filter', 'value::>=::100');
      const { setColumnIds } = renderColumnsHook();

      setColumnIds(['col2', 'col3']);

      expect(mockSearchParams.get('filter')).toBe('value::>=::100');
      expect(mockSearchParams.get('selectedColumns')).toBe('col2,col3');
    });

    it('preserves other params when removing', () => {
      mockSearchParams.set('filter', 'value::>=::100');
      mockSearchParams.set('selectedColumns', 'col1,col2');
      const { setColumnIds } = renderColumnsHook();

      setColumnIds(undefined);

      expect(mockSearchParams.get('filter')).toBe('value::>=::100');
      expect(mockSearchParams.has('selectedColumns')).toBe(false);
    });
  });
});
