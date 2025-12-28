import { describe, afterEach, test, jest, expect, beforeEach } from '@jest/globals';
import { renderHook, cleanup, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useEndpointQuery } from './useEndpointQuery';
import { GatewayApi } from '../api';
import type { Endpoint } from '../types';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

const generateMockEndpoint = (id: string): Endpoint => ({
  endpoint_id: `ep-${id}`,
  name: `endpoint-${id}`,
  created_at: 1700000000000,
  last_updated_at: 1700000000000 + 1000,
  model_mappings: [
    {
      mapping_id: `mm-${id}`,
      endpoint_id: `ep-${id}`,
      model_definition_id: `md-${id}`,
      weight: 1,
      created_at: 1700000000000,
      model_definition: {
        model_definition_id: `md-${id}`,
        name: `model-def-${id}`,
        provider: 'openai',
        model_name: 'gpt-4',
        secret_id: `secret-${id}`,
        secret_name: `secret-${id}`,
        created_at: 1700000000000,
        last_updated_at: 1700000000000,
        endpoint_count: 1,
      },
    },
  ],
});

describe('useEndpointQuery', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  test('fetches endpoint successfully', async () => {
    const mockEndpoint = generateMockEndpoint('1');

    jest.spyOn(GatewayApi, 'getEndpoint').mockResolvedValue({
      endpoint: mockEndpoint,
    });

    const { result } = renderHook(() => useEndpointQuery('ep-1'), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data?.endpoint).toEqual(mockEndpoint);
    expect(result.current.error).toBeNull();
    expect(GatewayApi.getEndpoint).toHaveBeenCalledWith('ep-1');
  });

  test('does not fetch when endpointId is empty', async () => {
    jest.spyOn(GatewayApi, 'getEndpoint').mockResolvedValue({
      endpoint: generateMockEndpoint('1'),
    });

    const { result } = renderHook(() => useEndpointQuery(''), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.fetchStatus).toBe('idle');
    });

    expect(GatewayApi.getEndpoint).not.toHaveBeenCalled();
    expect(result.current.data).toBeUndefined();
  });

  test('handles error state', async () => {
    jest.spyOn(GatewayApi, 'getEndpoint').mockRejectedValue(new Error('Endpoint not found'));

    const { result } = renderHook(() => useEndpointQuery('ep-invalid'), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBeInstanceOf(Error);
    expect((result.current.error as Error)?.message).toBe('Endpoint not found');
    expect(result.current.data).toBeUndefined();
  });
});
