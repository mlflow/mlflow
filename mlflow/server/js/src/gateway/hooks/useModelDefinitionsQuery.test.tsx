import { describe, afterEach, test, jest, expect, beforeEach } from '@jest/globals';
import { renderHook, cleanup, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useModelDefinitionsQuery } from './useModelDefinitionsQuery';
import { GatewayApi } from '../api';
import type { ModelDefinition } from '../types';

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

const generateMockModelDefinition = (id: string, provider = 'openai'): ModelDefinition => ({
  model_definition_id: `md-${id}`,
  name: `model-def-${id}`,
  secret_id: `secret-${id}`,
  secret_name: `secret-name-${id}`,
  provider,
  model_name: 'gpt-4',
  created_at: 1700000000000,
  last_updated_at: 1700000001000,
  endpoint_count: 1,
});

describe('useModelDefinitionsQuery', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  test('fetches model definitions successfully', async () => {
    const mockModelDefinitions = [generateMockModelDefinition('1'), generateMockModelDefinition('2')];

    jest.spyOn(GatewayApi, 'listModelDefinitions').mockResolvedValue({
      model_definitions: mockModelDefinitions,
    });

    const { result } = renderHook(() => useModelDefinitionsQuery(), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toHaveLength(2);
    expect(result.current.data?.[0].model_definition_id).toBe('md-1');
    expect(result.current.error).toBeUndefined();
  });

  test('returns undefined when no model definitions exist', async () => {
    jest.spyOn(GatewayApi, 'listModelDefinitions').mockResolvedValue({
      model_definitions: [],
    });

    const { result } = renderHook(() => useModelDefinitionsQuery(), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toEqual([]);
    expect(result.current.error).toBeUndefined();
  });

  test('handles error state', async () => {
    jest.spyOn(GatewayApi, 'listModelDefinitions').mockRejectedValue(new Error('Network error'));

    const { result } = renderHook(() => useModelDefinitionsQuery(), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBeInstanceOf(Error);
    expect(result.current.error?.message).toBe('Network error');
  });
});
