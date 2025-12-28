import { describe, afterEach, test, jest, expect, beforeEach } from '@jest/globals';
import { renderHook, cleanup, waitFor, act } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useUpdateEndpointMutation } from './useUpdateEndpointMutation';
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

describe('useUpdateEndpointMutation', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  test('updates endpoint name successfully', async () => {
    const mockResponse = {
      endpoint: {
        endpoint_id: 'ep-123',
        name: 'updated-endpoint',
        created_at: 1700000000000,
        last_updated_at: 1700000000001,
        model_mappings: [],
      },
    };

    jest.spyOn(GatewayApi, 'updateEndpoint').mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useUpdateEndpointMutation(), { wrapper: createWrapper() });

    await act(async () => {
      await result.current.mutateAsync({
        endpointId: 'ep-123',
        name: 'updated-endpoint',
      });
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(GatewayApi.updateEndpoint).toHaveBeenCalledWith({
      endpoint_id: 'ep-123',
      name: 'updated-endpoint',
    });
  });

  test('updates endpoint without name', async () => {
    const mockResponse = {
      endpoint: {
        endpoint_id: 'ep-123',
        name: 'endpoint-123',
        created_at: 1700000000000,
        last_updated_at: 1700000000001,
        model_mappings: [],
      },
    };

    jest.spyOn(GatewayApi, 'updateEndpoint').mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useUpdateEndpointMutation(), { wrapper: createWrapper() });

    await act(async () => {
      await result.current.mutateAsync({
        endpointId: 'ep-123',
      });
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(GatewayApi.updateEndpoint).toHaveBeenCalledWith({
      endpoint_id: 'ep-123',
      name: undefined,
    });
  });

  test('handles error state', async () => {
    const mockError = new Error('Update failed');
    jest.spyOn(GatewayApi, 'updateEndpoint').mockRejectedValue(mockError);

    const { result } = renderHook(() => useUpdateEndpointMutation(), { wrapper: createWrapper() });

    let caughtError: Error | undefined;
    await act(async () => {
      try {
        await result.current.mutateAsync({
          endpointId: 'ep-123',
          name: 'test',
        });
      } catch (e) {
        caughtError = e as Error;
      }
    });

    expect(caughtError?.message).toBe('Update failed');
    expect(GatewayApi.updateEndpoint).toHaveBeenCalled();
  });
});
