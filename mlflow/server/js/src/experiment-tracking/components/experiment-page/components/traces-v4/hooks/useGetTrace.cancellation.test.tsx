import React from 'react';
import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';

import { QueryClient, QueryClientProvider } from '../../../../../../common/utils/reactQueryHooks';
import { getTrace } from '../../../../../utils/TraceUtils';
import { useGetTrace } from './useGetTrace';
import {
  MOCK_V3_TRACE,
  MOCK_TRACE_INFO_V3,
} from '@databricks/web-shared/model-trace-explorer/ModelTraceExplorer.test-utils';

jest.mock('../../../../../utils/TraceUtils', () => ({ getTrace: jest.fn() }));
jest.mock('@databricks/web-shared/genai-traces-table', () => ({
  ...jest.requireActual<any>('@databricks/web-shared/genai-traces-table'),
  shouldUseTracesV4API: () => false,
  useFetchTraceV4LazyQuery: () => undefined,
}));

describe('drawer trace request cancellation', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });
  it('aborts the old transport when the selected trace changes', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    jest.mocked(getTrace).mockImplementation(() => new Promise(() => {}));
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
    const { rerender, unmount } = renderHook(({ id }) => useGetTrace(id), {
      initialProps: { id: 'tr-first' },
      wrapper,
    });
    try {
      await waitFor(() => expect(getTrace).toHaveBeenCalledTimes(1));
      const first = jest.mocked(getTrace).mock.calls[0][3];
      expect(first).toBeInstanceOf(AbortSignal);
      expect(first?.aborted).toBe(false);
      rerender({ id: 'tr-second' });
      await waitFor(() => expect(getTrace).toHaveBeenCalledTimes(2));
      expect(first?.aborted).toBe(true);
      const second = jest.mocked(getTrace).mock.calls[1][3];
      expect(second?.aborted).toBe(false);
      unmount();
      expect(second?.aborted).toBe(true);
    } finally {
      unmount();
      await act(async () => client.clear());
      jest.clearAllMocks();
    }
  });

  it('publishes shared requests once and ignores unrelated cache notifications', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    let finish!: (value: typeof MOCK_V3_TRACE) => void;
    jest.mocked(getTrace).mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
    let renders = 0;
    const { result, unmount } = renderHook(
      () => {
        renders++;
        return [useGetTrace('tr-shared'), useGetTrace('tr-shared')];
      },
      { wrapper },
    );
    try {
      await waitFor(() => expect(getTrace).toHaveBeenCalledTimes(1));
      expect(result.current.every((query) => query.isLoading && query.data === undefined)).toBe(true);
      await act(async () => finish(MOCK_V3_TRACE));
      await waitFor(() => expect(result.current.every((query) => query.isSuccess)).toBe(true));
      expect(result.current.every((query) => query.data === MOCK_V3_TRACE && !query.isLoading)).toBe(true);
      const settledRenders = renders;
      await act(async () => {
        for (let i = 0; i < 30; i++) client.setQueryData(['unrelated-row', i], { state: i });
      });
      expect(renders).toBe(settledRenders);
    } finally {
      unmount();
      await act(async () => client.clear());
    }
  });

  it('changes trace identity and loading together and ignores a stale completed response', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const pending = new Map<string, (value: typeof MOCK_V3_TRACE) => void>();
    jest.mocked(getTrace).mockImplementation(
      (id) =>
        new Promise((resolve) => {
          pending.set(id!, resolve);
        }),
    );
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
    const { result, rerender, unmount } = renderHook(({ id }) => useGetTrace(id), {
      initialProps: { id: 'tr-first' },
      wrapper,
    });
    try {
      await waitFor(() => expect(pending.has('tr-first')).toBe(true));
      rerender({ id: 'tr-second' });
      expect(result.current.data).toBeUndefined();
      expect(result.current.isLoading).toBe(true);
      await waitFor(() => expect(pending.has('tr-second')).toBe(true));
      const second = { ...MOCK_V3_TRACE, info: { ...MOCK_TRACE_INFO_V3, trace_id: 'tr-second' } };
      await act(async () => pending.get('tr-second')!(second));
      await waitFor(() => expect(result.current.data).toBe(second));
      await act(async () => pending.get('tr-first')!(MOCK_V3_TRACE));
      expect(result.current.data).toBe(second);
      expect(result.current.isLoading).toBe(false);
    } finally {
      unmount();
      await act(async () => client.clear());
    }
  });
});
