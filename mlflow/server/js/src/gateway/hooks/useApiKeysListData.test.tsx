import { describe, afterEach, test, jest, expect, beforeEach } from '@jest/globals';
import { renderHook, cleanup, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useApiKeysListData } from './useApiKeysListData';
import { GatewayApi } from '../api';
import type { SecretInfo, Endpoint, EndpointBinding, ModelDefinition } from '../types';

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
  last_updated_at: 1700000001000,
});

const generateMockEndpoint = (id: string, modelDefinitionSecretId?: string): Endpoint => ({
  endpoint_id: `ep-${id}`,
  name: `endpoint-${id}`,
  model_mappings: modelDefinitionSecretId
    ? [
        {
          mapping_id: `mapping-${id}`,
          endpoint_id: `ep-${id}`,
          model_definition_id: `md-${id}`,
          model_definition: {
            model_definition_id: `md-${id}`,
            name: `model-def-${id}`,
            secret_id: modelDefinitionSecretId,
            secret_name: `secret-name-${id}`,
            provider: 'openai',
            model_name: 'gpt-4',
            created_at: 1700000000000,
            last_updated_at: 1700000001000,
            endpoint_count: 1,
          },
          weight: 1,
          created_at: 1700000000000,
        },
      ]
    : [],
  created_at: 1700000000000,
  last_updated_at: 1700000001000,
});

const generateMockBinding = (id: string, endpointId: string): EndpointBinding => ({
  endpoint_id: endpointId,
  resource_type: 'experiment',
  resource_id: `exp-${id}`,
  created_at: 1700000000000,
});

const generateMockModelDefinition = (id: string, secretId: string): ModelDefinition => ({
  model_definition_id: `md-${id}`,
  name: `model-def-${id}`,
  secret_id: secretId,
  secret_name: `secret-name-${id}`,
  provider: 'openai',
  model_name: 'gpt-4',
  created_at: 1700000000000,
  last_updated_at: 1700000001000,
  endpoint_count: 1,
});

describe('useApiKeysListData', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  test('returns loading state initially', async () => {
    jest.spyOn(GatewayApi, 'listSecrets').mockImplementation(() => new Promise(() => {}));
    jest.spyOn(GatewayApi, 'listEndpoints').mockImplementation(() => new Promise(() => {}));
    jest.spyOn(GatewayApi, 'listModelDefinitions').mockImplementation(() => new Promise(() => {}));
    jest.spyOn(GatewayApi, 'listEndpointBindings').mockImplementation(() => new Promise(() => {}));

    const { result } = renderHook(() => useApiKeysListData({ searchFilter: '', filter: { providers: [] } }), {
      wrapper: createWrapper(),
    });

    expect(result.current.isLoading).toBe(true);
  });

  test('returns secrets and filters by search', async () => {
    const mockSecrets = [generateMockSecret('1', 'openai'), generateMockSecret('2', 'anthropic')];

    jest.spyOn(GatewayApi, 'listSecrets').mockResolvedValue({ secrets: mockSecrets });
    jest.spyOn(GatewayApi, 'listEndpoints').mockResolvedValue({ endpoints: [] });
    jest.spyOn(GatewayApi, 'listModelDefinitions').mockResolvedValue({ model_definitions: [] });
    jest.spyOn(GatewayApi, 'listEndpointBindings').mockResolvedValue({ bindings: [] });

    const { result, rerender } = renderHook(
      ({ searchFilter }) => useApiKeysListData({ searchFilter, filter: { providers: [] } }),
      { wrapper: createWrapper(), initialProps: { searchFilter: '' } },
    );

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.secrets).toHaveLength(2);
    expect(result.current.filteredSecrets).toHaveLength(2);

    rerender({ searchFilter: 'key-1' });
    expect(result.current.filteredSecrets).toHaveLength(1);
    expect(result.current.filteredSecrets[0].secret_id).toBe('secret-1');
  });

  test('filters by provider', async () => {
    const mockSecrets = [
      generateMockSecret('1', 'openai'),
      generateMockSecret('2', 'anthropic'),
      generateMockSecret('3', 'openai'),
    ];

    jest.spyOn(GatewayApi, 'listSecrets').mockResolvedValue({ secrets: mockSecrets });
    jest.spyOn(GatewayApi, 'listEndpoints').mockResolvedValue({ endpoints: [] });
    jest.spyOn(GatewayApi, 'listModelDefinitions').mockResolvedValue({ model_definitions: [] });
    jest.spyOn(GatewayApi, 'listEndpointBindings').mockResolvedValue({ bindings: [] });

    const { result, rerender } = renderHook(({ filter }) => useApiKeysListData({ searchFilter: '', filter }), {
      wrapper: createWrapper(),
      initialProps: { filter: { providers: [] as string[] } },
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.filteredSecrets).toHaveLength(3);

    rerender({ filter: { providers: ['openai'] } });
    expect(result.current.filteredSecrets).toHaveLength(2);
  });

  test('extracts available providers', async () => {
    const mockSecrets = [
      generateMockSecret('1', 'openai'),
      generateMockSecret('2', 'anthropic'),
      generateMockSecret('3', 'openai'),
    ];

    jest.spyOn(GatewayApi, 'listSecrets').mockResolvedValue({ secrets: mockSecrets });
    jest.spyOn(GatewayApi, 'listEndpoints').mockResolvedValue({ endpoints: [] });
    jest.spyOn(GatewayApi, 'listModelDefinitions').mockResolvedValue({ model_definitions: [] });
    jest.spyOn(GatewayApi, 'listEndpointBindings').mockResolvedValue({ bindings: [] });

    const { result } = renderHook(() => useApiKeysListData({ searchFilter: '', filter: { providers: [] } }), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.availableProviders).toEqual(expect.arrayContaining(['openai', 'anthropic']));
    expect(result.current.availableProviders).toHaveLength(2);
  });

  test('maps model definitions to secrets', async () => {
    const mockSecrets = [generateMockSecret('1', 'openai')];
    const mockModelDefinitions = [
      generateMockModelDefinition('md-1', 'secret-1'),
      generateMockModelDefinition('md-2', 'secret-1'),
    ];

    jest.spyOn(GatewayApi, 'listSecrets').mockResolvedValue({ secrets: mockSecrets });
    jest.spyOn(GatewayApi, 'listEndpoints').mockResolvedValue({ endpoints: [] });
    jest.spyOn(GatewayApi, 'listModelDefinitions').mockResolvedValue({ model_definitions: mockModelDefinitions });
    jest.spyOn(GatewayApi, 'listEndpointBindings').mockResolvedValue({ bindings: [] });

    const { result } = renderHook(() => useApiKeysListData({ searchFilter: '', filter: { providers: [] } }), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    const modelDefs = result.current.getModelDefinitionsForSecret('secret-1');
    expect(modelDefs).toHaveLength(2);
  });

  test('maps endpoints to secrets and counts them', async () => {
    const mockSecrets = [generateMockSecret('1', 'openai')];
    const mockEndpoints = [generateMockEndpoint('ep-1', 'secret-1'), generateMockEndpoint('ep-2', 'secret-1')];

    jest.spyOn(GatewayApi, 'listSecrets').mockResolvedValue({ secrets: mockSecrets });
    jest.spyOn(GatewayApi, 'listEndpoints').mockResolvedValue({ endpoints: mockEndpoints });
    jest.spyOn(GatewayApi, 'listModelDefinitions').mockResolvedValue({ model_definitions: [] });
    jest.spyOn(GatewayApi, 'listEndpointBindings').mockResolvedValue({ bindings: [] });

    const { result } = renderHook(() => useApiKeysListData({ searchFilter: '', filter: { providers: [] } }), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.getEndpointCount('secret-1')).toBe(2);
    const endpoints = result.current.getEndpointsForSecret('secret-1');
    expect(endpoints).toHaveLength(2);
  });

  test('maps bindings to secrets through endpoints', async () => {
    const mockSecrets = [generateMockSecret('1', 'openai')];
    const mockEndpoints = [generateMockEndpoint('1', 'secret-1')];
    const mockBindings = [generateMockBinding('b-1', 'ep-1'), generateMockBinding('b-2', 'ep-1')];

    jest.spyOn(GatewayApi, 'listSecrets').mockResolvedValue({ secrets: mockSecrets });
    jest.spyOn(GatewayApi, 'listEndpoints').mockResolvedValue({ endpoints: mockEndpoints });
    jest.spyOn(GatewayApi, 'listModelDefinitions').mockResolvedValue({ model_definitions: [] });
    jest.spyOn(GatewayApi, 'listEndpointBindings').mockResolvedValue({ bindings: mockBindings });

    const { result } = renderHook(() => useApiKeysListData({ searchFilter: '', filter: { providers: [] } }), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.getBindingCount('secret-1')).toBe(2);
    const bindings = result.current.getBindingsForSecret('secret-1');
    expect(bindings).toHaveLength(2);
  });

  test('returns empty arrays for secrets with no relationships', async () => {
    const mockSecrets = [generateMockSecret('1', 'openai')];

    jest.spyOn(GatewayApi, 'listSecrets').mockResolvedValue({ secrets: mockSecrets });
    jest.spyOn(GatewayApi, 'listEndpoints').mockResolvedValue({ endpoints: [] });
    jest.spyOn(GatewayApi, 'listModelDefinitions').mockResolvedValue({ model_definitions: [] });
    jest.spyOn(GatewayApi, 'listEndpointBindings').mockResolvedValue({ bindings: [] });

    const { result } = renderHook(() => useApiKeysListData({ searchFilter: '', filter: { providers: [] } }), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.getModelDefinitionsForSecret('secret-1')).toEqual([]);
    expect(result.current.getEndpointsForSecret('secret-1')).toEqual([]);
    expect(result.current.getBindingsForSecret('secret-1')).toEqual([]);
    expect(result.current.getEndpointCount('secret-1')).toBe(0);
    expect(result.current.getBindingCount('secret-1')).toBe(0);
  });
});
