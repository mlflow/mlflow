import { describe, afterEach, test, jest, expect, beforeEach } from '@jest/globals';
import { renderHook, cleanup, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useProviderConfigQuery } from './useProviderConfigQuery';
import { GatewayApi } from '../api';
import type { ProviderConfig } from '../types';

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

const mockProviderConfig: ProviderConfig = {
  auth_modes: [
    {
      mode: 'api_key',
      display_name: 'API Key',
      description: 'Authenticate using an API key',
      secret_fields: [{ name: 'api_key', type: 'string', required: true }],
      config_fields: [],
    },
  ],
  default_mode: 'api_key',
};

describe('useProviderConfigQuery', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  test('fetches provider config successfully', async () => {
    jest.spyOn(GatewayApi, 'getProviderConfig').mockResolvedValue(mockProviderConfig);

    const { result } = renderHook(() => useProviderConfigQuery({ provider: 'openai' }), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toEqual(mockProviderConfig);
    expect(result.current.error).toBeUndefined();
    expect(GatewayApi.getProviderConfig).toHaveBeenCalledWith('openai');
  });

  test('does not fetch when provider is empty string', async () => {
    jest.spyOn(GatewayApi, 'getProviderConfig').mockResolvedValue(mockProviderConfig);

    renderHook(() => useProviderConfigQuery({ provider: '' }), { wrapper: createWrapper() });

    expect(GatewayApi.getProviderConfig).not.toHaveBeenCalled();
  });

  test('fetches config for different providers', async () => {
    const anthropicConfig: ProviderConfig = {
      auth_modes: [
        {
          mode: 'api_key',
          display_name: 'API Key',
          secret_fields: [{ name: 'api_key', type: 'string', required: true }],
          config_fields: [],
        },
      ],
      default_mode: 'api_key',
    };

    jest.spyOn(GatewayApi, 'getProviderConfig').mockResolvedValue(anthropicConfig);

    const { result } = renderHook(() => useProviderConfigQuery({ provider: 'anthropic' }), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toEqual(anthropicConfig);
    expect(GatewayApi.getProviderConfig).toHaveBeenCalledWith('anthropic');
  });

  test('handles error state', async () => {
    jest.spyOn(GatewayApi, 'getProviderConfig').mockRejectedValue(new Error('Provider not supported'));

    const { result } = renderHook(() => useProviderConfigQuery({ provider: 'invalid' }), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBeInstanceOf(Error);
    expect(result.current.error?.message).toBe('Provider not supported');
  });

  test('provides refetch function', async () => {
    jest.spyOn(GatewayApi, 'getProviderConfig').mockResolvedValue(mockProviderConfig);

    const { result } = renderHook(() => useProviderConfigQuery({ provider: 'openai' }), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(typeof result.current.refetch).toBe('function');
  });

  test('returns config with multiple auth modes', async () => {
    const multiModeConfig: ProviderConfig = {
      auth_modes: [
        {
          mode: 'api_key',
          display_name: 'API Key',
          secret_fields: [{ name: 'api_key', type: 'string', required: true }],
          config_fields: [],
        },
        {
          mode: 'iam',
          display_name: 'IAM Role',
          description: 'Authenticate using AWS IAM',
          secret_fields: [
            { name: 'aws_access_key_id', type: 'string', required: true },
            { name: 'aws_secret_access_key', type: 'string', required: true },
          ],
          config_fields: [{ name: 'region', type: 'string', required: true }],
        },
      ],
      default_mode: 'api_key',
    };

    jest.spyOn(GatewayApi, 'getProviderConfig').mockResolvedValue(multiModeConfig);

    const { result } = renderHook(() => useProviderConfigQuery({ provider: 'bedrock' }), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data?.auth_modes).toHaveLength(2);
    expect(result.current.data?.default_mode).toBe('api_key');
  });
});
