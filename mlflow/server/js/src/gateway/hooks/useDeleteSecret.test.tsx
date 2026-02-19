import { describe, afterEach, test, jest, expect, beforeEach } from '@jest/globals';
import { renderHook, cleanup, waitFor, act } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useDeleteSecret } from './useDeleteSecret';
import { GatewayApi } from '../api';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

describe('useDeleteSecret', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  test('deletes secret successfully', async () => {
    jest.spyOn(GatewayApi, 'deleteSecret').mockResolvedValue(undefined);

    const { result } = renderHook(() => useDeleteSecret(), { wrapper: createWrapper() });

    await act(async () => {
      await result.current.mutateAsync('secret-123');
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(GatewayApi.deleteSecret).toHaveBeenCalledWith('secret-123');
  });

  test('handles error state', async () => {
    jest.spyOn(GatewayApi, 'deleteSecret').mockRejectedValue(new Error('Secret not found'));

    const { result } = renderHook(() => useDeleteSecret(), { wrapper: createWrapper() });

    let caughtError: Error | undefined;
    await act(async () => {
      try {
        await result.current.mutateAsync('nonexistent');
      } catch (e) {
        caughtError = e as Error;
      }
    });

    expect(caughtError?.message).toBe('Secret not found');
    expect(GatewayApi.deleteSecret).toHaveBeenCalled();
  });
});
