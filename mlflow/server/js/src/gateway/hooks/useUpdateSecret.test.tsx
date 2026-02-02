import { describe, afterEach, test, jest, expect, beforeEach } from '@jest/globals';
import { renderHook, cleanup, waitFor, act } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useUpdateSecret } from './useUpdateSecret';
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

describe('useUpdateSecret', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  test('updates secret successfully', async () => {
    const mockResponse = {
      secret: {
        secret_id: 'secret-123',
        secret_name: 'my-api-key',
        masked_values: { api_key: 'sk-...new' },
        provider: 'openai',
        created_at: 1700000000000,
        last_updated_at: 1700001000000,
      },
    };

    jest.spyOn(GatewayApi, 'updateSecret').mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useUpdateSecret(), { wrapper: createWrapper() });

    await act(async () => {
      await result.current.mutateAsync({
        secret_id: 'secret-123',
        secret_value: { api_key: 'sk-new-key' },
      });
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(GatewayApi.updateSecret).toHaveBeenCalledWith({
      secret_id: 'secret-123',
      secret_value: { api_key: 'sk-new-key' },
    });
  });

  test('handles error state', async () => {
    jest.spyOn(GatewayApi, 'updateSecret').mockRejectedValue(new Error('Secret not found'));

    const { result } = renderHook(() => useUpdateSecret(), { wrapper: createWrapper() });

    let caughtError: Error | undefined;
    await act(async () => {
      try {
        await result.current.mutateAsync({
          secret_id: 'nonexistent',
          secret_value: { api_key: 'sk-key' },
        });
      } catch (e) {
        caughtError = e as Error;
      }
    });

    expect(caughtError?.message).toBe('Secret not found');
    expect(GatewayApi.updateSecret).toHaveBeenCalled();
  });
});
