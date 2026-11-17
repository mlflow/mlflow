import React from 'react';
import { describe, expect, it, jest } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';

import { QueryClient, QueryClientProvider } from '../../../../../../common/utils/reactQueryHooks';
import { getTrace } from '../../../../../utils/TraceUtils';
import { useGetTrace } from './useGetTrace';

jest.mock('../../../../../utils/TraceUtils', () => ({ getTrace: jest.fn() }));
jest.mock('@databricks/web-shared/genai-traces-table', () => ({
  ...jest.requireActual<any>('@databricks/web-shared/genai-traces-table'),
  shouldUseTracesV4API: () => false,
  useFetchTraceV4LazyQuery: () => undefined,
}));

describe('drawer trace request cancellation', () => {
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
});
