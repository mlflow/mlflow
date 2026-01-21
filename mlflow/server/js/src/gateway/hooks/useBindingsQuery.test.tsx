import { describe, afterEach, test, jest, expect, beforeEach } from '@jest/globals';
import { renderHook, cleanup, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useBindingsQuery } from './useBindingsQuery';
import { GatewayApi } from '../api';
import type { EndpointBinding } from '../types';

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

const generateMockBinding = (id: string): EndpointBinding => ({
  endpoint_id: `ep-${id}`,
  resource_type: 'scorer',
  resource_id: `job-${id}`,
  created_at: 1700000000000,
  display_name: `Scorer ${id}`,
});

describe('useBindingsQuery', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  test('fetches bindings successfully', async () => {
    const mockBindings = [generateMockBinding('1'), generateMockBinding('2')];

    jest.spyOn(GatewayApi, 'listEndpointBindings').mockResolvedValue({
      bindings: mockBindings,
    });

    const { result } = renderHook(() => useBindingsQuery(), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toHaveLength(2);
    expect(result.current.data[0].endpoint_id).toBe('ep-1');
    expect(result.current.error).toBeUndefined();
  });

  test('returns empty array when no bindings exist', async () => {
    jest.spyOn(GatewayApi, 'listEndpointBindings').mockResolvedValue({
      bindings: [],
    });

    const { result } = renderHook(() => useBindingsQuery(), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toEqual([]);
    expect(result.current.error).toBeUndefined();
  });

  test('handles error state', async () => {
    jest.spyOn(GatewayApi, 'listEndpointBindings').mockRejectedValue(new Error('Network error'));

    const { result } = renderHook(() => useBindingsQuery(), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBeInstanceOf(Error);
    expect(result.current.error?.message).toBe('Network error');
    expect(result.current.data).toEqual([]);
  });
});
