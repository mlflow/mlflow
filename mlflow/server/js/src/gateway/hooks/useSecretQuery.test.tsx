import { describe, afterEach, test, jest, expect, beforeEach } from '@jest/globals';
import { renderHook, cleanup, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useSecretQuery } from './useSecretQuery';
import { GatewayApi } from '../api';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

describe('useSecretQuery', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  test('fetches secret by ID successfully', async () => {
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

    jest.spyOn(GatewayApi, 'getSecret').mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useSecretQuery('secret-123'), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(GatewayApi.getSecret).toHaveBeenCalledWith('secret-123');
    expect(result.current.data?.secret.secret_id).toBe('secret-123');
    expect(result.current.error).toBeFalsy();
  });

  test('does not fetch when secretId is undefined', () => {
    jest.spyOn(GatewayApi, 'getSecret');

    const { result } = renderHook(() => useSecretQuery(undefined), { wrapper: createWrapper() });

    expect(result.current.isFetching).toBe(false);
    expect(GatewayApi.getSecret).not.toHaveBeenCalled();
    expect(result.current.data).toBeUndefined();
  });

  test('handles error state', async () => {
    jest.spyOn(GatewayApi, 'getSecret').mockRejectedValue(new Error('Secret not found'));

    const { result } = renderHook(() => useSecretQuery('nonexistent'), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBeInstanceOf(Error);
    expect((result.current.error as Error).message).toBe('Secret not found');
  });
});
