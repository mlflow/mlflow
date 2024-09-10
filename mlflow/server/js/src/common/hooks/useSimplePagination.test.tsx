import { act, renderHook } from '@testing-library/react';
import { useSimplePagination } from './useSimplePagination';

describe('useSimplePagination', () => {
  it('should initialize with the correct default values', () => {
    const { result } = renderHook(() => useSimplePagination(10));

    expect(result.current.pageIndex).toBe(1);
    expect(result.current.pageSize).toBe(10);
  });

  it('should update the page size correctly', () => {
    const { result } = renderHook(() => useSimplePagination(10));

    act(() => {
      result.current.setPageSize(20);
    });

    expect(result.current.pageSize).toBe(20);
    expect(result.current.pageIndex).toBe(1);
  });

  it('should update the current page index correctly', () => {
    const { result } = renderHook(() => useSimplePagination(10));

    act(() => {
      result.current.setCurrentPageIndex(2);
    });

    expect(result.current.pageIndex).toBe(2);
    expect(result.current.pageSize).toBe(10);

    // Assert resetting the page size
    act(() => {
      result.current.setPageSize(20);
    });

    expect(result.current.pageSize).toBe(20);
    expect(result.current.pageIndex).toBe(1);
  });
});
