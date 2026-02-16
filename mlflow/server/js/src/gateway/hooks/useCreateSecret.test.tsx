import { describe, afterEach, test, jest, expect, beforeEach } from '@jest/globals';
import { renderHook, cleanup, waitFor, act } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useCreateSecret } from './useCreateSecret';
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

describe('useCreateSecret', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  test('creates secret successfully', async () => {
    const mockResponse = {
      secret: {
        secret_id: 'secret-123',
        secret_name: 'my-api-key',
        masked_values: { api_key: 'sk-...xxx' },
        provider: 'openai',
        created_at: 1700000000000,
        last_updated_at: 1700000000000,
      },
    };

    jest.spyOn(GatewayApi, 'createSecret').mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useCreateSecret(), { wrapper: createWrapper() });

    await act(async () => {
      await result.current.mutateAsync({
        secret_name: 'my-api-key',
        secret_value: { api_key: 'sk-real-key' },
        provider: 'openai',
      });
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(GatewayApi.createSecret).toHaveBeenCalledWith({
      secret_name: 'my-api-key',
      secret_value: { api_key: 'sk-real-key' },
      provider: 'openai',
    });
  });

  test('handles error state', async () => {
    const mockError = new Error('Duplicate key name');
    jest.spyOn(GatewayApi, 'createSecret').mockRejectedValue(mockError);

    const { result } = renderHook(() => useCreateSecret(), { wrapper: createWrapper() });

    let caughtError: Error | undefined;
    await act(async () => {
      try {
        await result.current.mutateAsync({
          secret_name: 'existing-key',
          secret_value: { api_key: 'sk-key' },
          provider: 'openai',
        });
      } catch (e) {
        caughtError = e as Error;
      }
    });

    expect(caughtError?.message).toBe('Duplicate key name');
    expect(GatewayApi.createSecret).toHaveBeenCalled();
  });
});
