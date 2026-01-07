import { describe, afterEach, test, jest, expect, beforeEach } from '@jest/globals';
import { renderHook, cleanup, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useProvidersQuery } from './useProvidersQuery';
import { GatewayApi } from '../api';

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

describe('useProvidersQuery', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  test('fetches providers successfully', async () => {
    const mockProviders = ['openai', 'anthropic', 'cohere'];

    jest.spyOn(GatewayApi, 'listProviders').mockResolvedValue({
      providers: mockProviders,
    });

    const { result } = renderHook(() => useProvidersQuery(), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toEqual(mockProviders);
    expect(result.current.error).toBeUndefined();
    expect(GatewayApi.listProviders).toHaveBeenCalledTimes(1);
  });

  test('returns empty array when no providers exist', async () => {
    jest.spyOn(GatewayApi, 'listProviders').mockResolvedValue({
      providers: [],
    });

    const { result } = renderHook(() => useProvidersQuery(), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toEqual([]);
    expect(result.current.error).toBeUndefined();
  });

  test('handles error state', async () => {
    jest.spyOn(GatewayApi, 'listProviders').mockRejectedValue(new Error('Network error'));

    const { result } = renderHook(() => useProvidersQuery(), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBeInstanceOf(Error);
    expect(result.current.error?.message).toBe('Network error');
  });

  test('provides refetch function', async () => {
    jest.spyOn(GatewayApi, 'listProviders').mockResolvedValue({
      providers: ['openai'],
    });

    const { result } = renderHook(() => useProvidersQuery(), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(typeof result.current.refetch).toBe('function');
  });
});
