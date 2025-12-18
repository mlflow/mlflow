import { describe, afterEach, test, jest, expect, beforeEach } from '@jest/globals';
import { renderHook, cleanup, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useModelsQuery } from './useModelsQuery';
import { GatewayApi } from '../api';
import type { Model } from '../types';

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

const mockModels: Model[] = [
  { model: 'gpt-4', provider: 'openai', supports_function_calling: true },
  { model: 'gpt-3.5-turbo', provider: 'openai', supports_function_calling: true },
];

describe('useModelsQuery', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  test('fetches models for a provider successfully', async () => {
    jest.spyOn(GatewayApi, 'listModels').mockResolvedValue({
      models: mockModels,
    });

    const { result } = renderHook(() => useModelsQuery({ provider: 'openai' }), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toEqual(mockModels);
    expect(result.current.error).toBeUndefined();
    expect(GatewayApi.listModels).toHaveBeenCalledWith('openai');
  });

  test('does not fetch when provider is undefined', async () => {
    jest.spyOn(GatewayApi, 'listModels').mockResolvedValue({
      models: mockModels,
    });

    renderHook(() => useModelsQuery({ provider: undefined }), { wrapper: createWrapper() });

    expect(GatewayApi.listModels).not.toHaveBeenCalled();
  });

  test('does not fetch when called with no arguments', async () => {
    jest.spyOn(GatewayApi, 'listModels').mockResolvedValue({
      models: mockModels,
    });

    renderHook(() => useModelsQuery(), { wrapper: createWrapper() });

    expect(GatewayApi.listModels).not.toHaveBeenCalled();
  });

  test('returns empty array when no models exist', async () => {
    jest.spyOn(GatewayApi, 'listModels').mockResolvedValue({
      models: [],
    });

    const { result } = renderHook(() => useModelsQuery({ provider: 'openai' }), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toEqual([]);
    expect(result.current.error).toBeUndefined();
  });

  test('handles error state', async () => {
    jest.spyOn(GatewayApi, 'listModels').mockRejectedValue(new Error('Provider not found'));

    const { result } = renderHook(() => useModelsQuery({ provider: 'invalid' }), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBeInstanceOf(Error);
    expect(result.current.error?.message).toBe('Provider not found');
  });

  test('provides refetch function', async () => {
    jest.spyOn(GatewayApi, 'listModels').mockResolvedValue({
      models: mockModels,
    });

    const { result } = renderHook(() => useModelsQuery({ provider: 'openai' }), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(typeof result.current.refetch).toBe('function');
  });
});
