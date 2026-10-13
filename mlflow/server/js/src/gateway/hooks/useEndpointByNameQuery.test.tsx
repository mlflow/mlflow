import { describe, afterEach, test, jest, expect, beforeEach } from '@jest/globals';
import { renderHook, cleanup, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { useEndpointByNameQuery } from './useEndpointByNameQuery';
import * as FetchUtils from '../../common/utils/FetchUtils';
import type { Endpoint } from '../types';

jest.mock('../../common/utils/FetchUtils', () => ({
  fetchAPI: jest.fn(),
  getAjaxUrl: jest.fn((url: string) => url),
}));

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

const generateMockEndpoint = (name: string): Endpoint => ({
  endpoint_id: `ep-${name}`,
  name,
  created_at: 1700000000000,
  last_updated_at: 1700000000000 + 1000,
  model_mappings: [
    {
      mapping_id: `mm-${name}`,
      endpoint_id: `ep-${name}`,
      model_definition_id: `md-${name}`,
      weight: 1,
      created_at: 1700000000000,
      model_definition: {
        model_definition_id: `md-${name}`,
        name: `model-def-${name}`,
        provider: 'openai',
        model_name: 'gpt-4',
        secret_id: `secret-${name}`,
        secret_name: `secret-${name}`,
        created_at: 1700000000000,
        last_updated_at: 1700000000000,
        endpoint_count: 1,
      },
    },
  ],
});

describe('useEndpointByNameQuery', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  test('fetches endpoint by name successfully', async () => {
    const mockEndpoint = generateMockEndpoint('my-endpoint');

    jest.mocked(FetchUtils.fetchAPI).mockResolvedValue({
      endpoint: mockEndpoint,
    });

    const { result } = renderHook(() => useEndpointByNameQuery('my-endpoint'), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data?.endpoint).toEqual(mockEndpoint);
    expect(result.current.error).toBeNull();
    expect(FetchUtils.fetchAPI).toHaveBeenCalledWith(
      expect.stringContaining('ajax-api/3.0/mlflow/gateway/endpoints/get?name=my-endpoint'),
    );
  });

  test('does not fetch when name is undefined', async () => {
    const { result } = renderHook(() => useEndpointByNameQuery(undefined), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.fetchStatus).toBe('idle');
    });

    expect(FetchUtils.fetchAPI).not.toHaveBeenCalled();
    expect(result.current.data).toBeUndefined();
  });

  test('does not fetch when name is empty string', async () => {
    const { result } = renderHook(() => useEndpointByNameQuery(''), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.fetchStatus).toBe('idle');
    });

    expect(FetchUtils.fetchAPI).not.toHaveBeenCalled();
    expect(result.current.data).toBeUndefined();
  });

  test('handles error state when endpoint not found', async () => {
    jest.mocked(FetchUtils.fetchAPI).mockRejectedValue(new Error('Endpoint not found'));

    const { result } = renderHook(() => useEndpointByNameQuery('nonexistent-endpoint'), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBeInstanceOf(Error);
    expect((result.current.error as Error)?.message).toBe('Endpoint not found');
    expect(result.current.data).toBeUndefined();
  });
});
