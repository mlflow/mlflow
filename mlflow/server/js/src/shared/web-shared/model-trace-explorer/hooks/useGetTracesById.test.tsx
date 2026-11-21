import { describe, expect, it, jest } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';

import { QueryClient, QueryClientProvider } from '../../query-client/queryClient';
import { TracesServiceV3 } from '../api';
import { GET_TRACE_BY_ID, useGetTracesById } from './useGetTracesById';

jest.mock('../api', () => ({ TracesServiceV3: { getTraceV3: jest.fn() } }));

describe('comparison trace query cancellation', () => {
  it('passes signals to both traces and aborts on cancellation and unmount', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    jest.mocked(TracesServiceV3.getTraceV3).mockImplementation(() => new Promise(() => {}));
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
    const { unmount } = renderHook(() => useGetTracesById(['tr-first', 'tr-second']), { wrapper });
    try {
      await waitFor(() => expect(TracesServiceV3.getTraceV3).toHaveBeenCalledTimes(2));
      const [first, second] = jest.mocked(TracesServiceV3.getTraceV3).mock.calls.map(([, signal]) => signal);
      expect(first).toBeInstanceOf(AbortSignal);
      expect(second).toBeInstanceOf(AbortSignal);
      expect(first?.aborted).toBe(false);
      expect(second?.aborted).toBe(false);
      await act(async () => {
        await client.cancelQueries([GET_TRACE_BY_ID, 'tr-first']);
      });
      expect(first?.aborted).toBe(true);
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
