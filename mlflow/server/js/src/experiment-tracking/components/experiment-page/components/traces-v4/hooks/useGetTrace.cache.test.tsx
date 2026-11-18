import React from 'react';
import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';

import { QueryClient, QueryClientProvider } from '../../../../../../common/utils/reactQueryHooks';
import {
  MOCK_V3_TRACE,
  MOCK_V3_SPANS,
} from '@databricks/web-shared/model-trace-explorer/ModelTraceExplorer.test-utils';
import { getTrace } from '../../../../../utils/TraceUtils';
import { useGetTrace } from './useGetTrace';

jest.mock('../../../../../utils/TraceUtils', () => ({ getTrace: jest.fn() }));
jest.mock('@databricks/web-shared/genai-traces-table', () => ({
  ...jest.requireActual<any>('@databricks/web-shared/genai-traces-table'),
  shouldUseTracesV4API: () => false,
  useFetchTraceV4LazyQuery: () => undefined,
}));

describe('closed drawer trace retention', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it('keeps at most five inactive full traces while preserving the active result', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    jest.mocked(getTrace).mockResolvedValue(MOCK_V3_TRACE);
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
    const { result, rerender, unmount } = renderHook(({ id }) => useGetTrace(id), {
      initialProps: { id: 'tr-0' },
      wrapper,
    });
    try {
      for (let index = 0; index < 8; index++) {
        rerender({ id: `tr-${index}` });
        await waitFor(() => expect(result.current.isSuccess).toBe(true));
      }
      expect(client.getQueryCache().findAll({ queryKey: ['getTrace'], type: 'inactive' })).toHaveLength(5);
      expect(result.current.data).toEqual(MOCK_V3_TRACE);
      expect(client.getQueryData(['getTrace', 'tr-0', undefined])).toBeUndefined();
      expect(client.getQueryData(['getTrace', 'tr-7', undefined])).toEqual(MOCK_V3_TRACE);
      expect(getTrace).toHaveBeenCalledWith('tr-7', undefined, undefined, {
        signal: expect.any(AbortSignal),
        compactPayloads: true,
      });
    } finally {
      unmount();
      await act(async () => client.clear());
    }
  });

  it('displays an oversized trace but releases it after the drawer closes', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const trace = {
      ...MOCK_V3_TRACE,
      data: { spans: [{ ...MOCK_V3_SPANS[0], attributes: { payload: 'a'.repeat(64 * 1024 * 1024 + 1) } }] },
    };
    jest.mocked(getTrace).mockResolvedValue(trace);
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
    const { result, unmount } = renderHook(() => useGetTrace('tr-large'), { wrapper });
    try {
      await waitFor(() => expect(result.current.isSuccess).toBe(true));
      expect(result.current.data).toBe(trace);
      unmount();
      await act(async () => {});
      expect(client.getQueryData(['getTrace', 'tr-large', undefined])).toBeUndefined();
    } finally {
      unmount();
      await act(async () => client.clear());
    }
  });
});
