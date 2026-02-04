import { describe, afterEach, test, jest, expect, beforeEach } from '@jest/globals';
import { renderHook, cleanup, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useEndpointsQuery } from './useEndpointsQuery';
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

const generateMockEndpoint = (id: string, provider = 'openai'): Endpoint => ({
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
        provider,
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

describe('useEndpointsQuery', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  test('fetches endpoints successfully', async () => {
    const mockEndpoints = [generateMockEndpoint('1'), generateMockEndpoint('2'), generateMockEndpoint('3')];

    jest.spyOn(GatewayApi, 'listEndpoints').mockResolvedValue({
      endpoints: mockEndpoints,
    });

    const { result } = renderHook(() => useEndpointsQuery(), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toHaveLength(3);
    expect(result.current.data[0].endpoint_id).toBe('ep-1');
    expect(result.current.data[1].endpoint_id).toBe('ep-2');
    expect(result.current.error).toBeUndefined();
  });

  test('returns empty array when no endpoints exist', async () => {
    jest.spyOn(GatewayApi, 'listEndpoints').mockResolvedValue({
      endpoints: [],
    });

    const { result } = renderHook(() => useEndpointsQuery(), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toEqual([]);
    expect(result.current.error).toBeUndefined();
  });

  test('filters endpoints by provider', async () => {
    jest.spyOn(GatewayApi, 'listEndpoints').mockResolvedValue({
      endpoints: [generateMockEndpoint('1', 'anthropic')],
    });

    const { result } = renderHook(() => useEndpointsQuery({ provider: 'anthropic' }), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(GatewayApi.listEndpoints).toHaveBeenCalledWith('anthropic');
  });

  test('handles error state', async () => {
    jest.spyOn(GatewayApi, 'listEndpoints').mockRejectedValue(new Error('Network error'));

    const { result } = renderHook(() => useEndpointsQuery(), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBeInstanceOf(Error);
    expect(result.current.error?.message).toBe('Network error');
    expect(result.current.data).toEqual([]);
  });
});
