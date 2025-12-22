import { describe, afterEach, test, jest, expect, beforeEach } from '@jest/globals';
import { renderHook, cleanup, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useSecretsQuery } from './useSecretsQuery';
import { GatewayApi } from '../api';
import type { SecretInfo } from '../types';

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

const generateMockSecret = (id: string, provider = 'openai'): SecretInfo => ({
  secret_id: `secret-${id}`,
  secret_name: `api-key-${id}`,
  masked_values: { api_key: `sk-...${id}xx` },
  provider,
  created_at: 1700000000000,
  last_updated_at: 1700000000000 + 1000,
});

describe('useSecretsQuery', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  test('fetches secrets successfully', async () => {
    const mockSecrets = [generateMockSecret('1'), generateMockSecret('2')];

    jest.spyOn(GatewayApi, 'listSecrets').mockResolvedValue({
      secrets: mockSecrets,
    });

    const { result } = renderHook(() => useSecretsQuery(), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toHaveLength(2);
    expect(result.current.data[0].secret_id).toBe('secret-1');
    expect(result.current.error).toBeUndefined();
  });

  test('returns empty array when no secrets exist', async () => {
    jest.spyOn(GatewayApi, 'listSecrets').mockResolvedValue({
      secrets: [],
    });

    const { result } = renderHook(() => useSecretsQuery(), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toEqual([]);
    expect(result.current.error).toBeUndefined();
  });

  test('filters secrets by provider', async () => {
    jest.spyOn(GatewayApi, 'listSecrets').mockResolvedValue({
      secrets: [generateMockSecret('1', 'anthropic')],
    });

    const { result } = renderHook(() => useSecretsQuery({ provider: 'anthropic' }), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(GatewayApi.listSecrets).toHaveBeenCalledWith('anthropic');
  });

  test('handles error state', async () => {
    jest.spyOn(GatewayApi, 'listSecrets').mockRejectedValue(new Error('Unauthorized'));

    const { result } = renderHook(() => useSecretsQuery(), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBeInstanceOf(Error);
    expect(result.current.error?.message).toBe('Unauthorized');
    expect(result.current.data).toEqual([]);
  });
});
