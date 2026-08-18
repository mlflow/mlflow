import { describe, it, expect, jest, beforeEach, afterEach } from '@jest/globals';
import { renderHook, act } from '@testing-library/react';

import { useServerSideTraceSearch } from './useServerSideTraceSearch';
import { getExperimentTraceV3 } from '../api';
import type { ModelTraceInfoV3 } from '../ModelTrace.types';

jest.mock('../api', () => ({
  getExperimentTraceV3: jest.fn(),
}));

jest.mock('../ModelTraceExplorer.utils', () => ({
  isV3ModelTraceInfo: (info: any) => Boolean(info && 'trace_location' in info),
  parseModelTraceToTreeWithMultipleRoots: jest.fn((trace: any) =>
    (trace?.data?.spans ?? []).map((span: any) => ({
      key: span.span_id,
      title: span.name,
    })),
  ),
}));

const MOCK_TRACE_INFO: ModelTraceInfoV3 = {
  trace_id: 'tr-test-123',
  trace_location: {
    type: 'MLFLOW_EXPERIMENT',
    mlflow_experiment: { experiment_id: '1' },
  },
  request_time: '2025-01-01',
  state: 'OK',
  tags: {},
};

describe('useServerSideTraceSearch', () => {
  const mockGetExperimentTraceV3 = jest.mocked(getExperimentTraceV3);

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('returns null filteredTreeNodes for empty filter', () => {
    const { result } = renderHook(() => useServerSideTraceSearch({ modelTraceInfo: MOCK_TRACE_INFO, enabled: true }));

    expect(result.current.searchFilter).toBe('');
    expect(result.current.filteredTreeNodes).toBeNull();
    expect(result.current.isSearching).toBe(false);
  });

  it('calls getExperimentTraceV3 with non-empty filter after debounce', async () => {
    mockGetExperimentTraceV3.mockResolvedValue({
      trace: {
        trace_info: MOCK_TRACE_INFO,
        spans: [{ span_id: 's1', name: 'matched-span' }],
      },
    } as any);

    const { result } = renderHook(() => useServerSideTraceSearch({ modelTraceInfo: MOCK_TRACE_INFO, enabled: true }));

    act(() => {
      result.current.setSearchFilter('search_text');
    });

    await act(async () => {
      jest.advanceTimersByTime(500);
    });

    expect(mockGetExperimentTraceV3).toHaveBeenCalledWith({
      traceId: 'tr-test-123',
      filter: 'search_text',
    });
    expect(result.current.filteredTreeNodes).toEqual([{ key: 's1', title: 'matched-span' }]);
  });

  it('debounces search and calls API only once for rapid filter changes', async () => {
    mockGetExperimentTraceV3.mockResolvedValue({
      trace: { trace_info: MOCK_TRACE_INFO, spans: [] },
    } as any);

    const { result } = renderHook(() => useServerSideTraceSearch({ modelTraceInfo: MOCK_TRACE_INFO, enabled: true }));

    act(() => result.current.setSearchFilter('a'));
    act(() => jest.advanceTimersByTime(100));
    act(() => result.current.setSearchFilter('ab'));
    act(() => jest.advanceTimersByTime(100));
    act(() => result.current.setSearchFilter('abc'));

    await act(async () => {
      jest.advanceTimersByTime(500);
    });

    expect(mockGetExperimentTraceV3).toHaveBeenCalledTimes(1);
    expect(mockGetExperimentTraceV3).toHaveBeenCalledWith({
      traceId: 'tr-test-123',
      filter: 'abc',
    });
  });

  it('clears isSearching and filteredTreeNodes when filter is set back to empty', async () => {
    mockGetExperimentTraceV3.mockResolvedValue({
      trace: { trace_info: MOCK_TRACE_INFO, spans: [{ span_id: 's1', name: 'span' }] },
    } as any);

    const { result } = renderHook(() => useServerSideTraceSearch({ modelTraceInfo: MOCK_TRACE_INFO, enabled: true }));

    act(() => result.current.setSearchFilter('text'));
    await act(async () => {
      jest.advanceTimersByTime(500);
    });

    expect(result.current.filteredTreeNodes).not.toBeNull();

    act(() => result.current.setSearchFilter(''));
    await act(async () => {
      jest.advanceTimersByTime(500);
    });

    expect(result.current.isSearching).toBe(false);
    expect(result.current.filteredTreeNodes).toBeNull();
  });

  it('ignores stale responses when a newer search is triggered', async () => {
    let resolveFirst!: (value: any) => void;
    let resolveSecond!: (value: any) => void;

    const firstPromise = new Promise((resolve) => {
      resolveFirst = resolve;
    });
    const secondPromise = new Promise((resolve) => {
      resolveSecond = resolve;
    });

    mockGetExperimentTraceV3.mockReturnValueOnce(firstPromise as any).mockReturnValueOnce(secondPromise as any);

    const { result } = renderHook(() => useServerSideTraceSearch({ modelTraceInfo: MOCK_TRACE_INFO, enabled: true }));

    act(() => result.current.setSearchFilter('searchA'));
    await act(async () => {
      jest.advanceTimersByTime(500);
    });

    act(() => result.current.setSearchFilter('searchB'));
    await act(async () => {
      jest.advanceTimersByTime(500);
    });

    await act(async () => {
      resolveSecond({
        trace: {
          trace_info: MOCK_TRACE_INFO,
          spans: [{ span_id: 'b1', name: 'result-B' }],
        },
      });
    });

    await act(async () => {
      resolveFirst({
        trace: {
          trace_info: MOCK_TRACE_INFO,
          spans: [{ span_id: 'a1', name: 'result-A' }],
        },
      });
    });

    expect(result.current.filteredTreeNodes).toEqual([{ key: 'b1', title: 'result-B' }]);
  });
});
