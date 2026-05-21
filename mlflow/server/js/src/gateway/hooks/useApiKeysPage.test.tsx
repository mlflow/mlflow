import { describe, afterEach, test, jest, expect, beforeEach } from '@jest/globals';
import { renderHook, cleanup, act } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { useApiKeysPage } from './useApiKeysPage';
import type { SecretInfo, Endpoint, EndpointBinding, ModelDefinition } from '../types';

const mockRefetchSecrets = jest.fn<() => Promise<{ data: { secrets: SecretInfo[] } }>>();
const mockRefetchEndpoints = jest.fn<() => Promise<{ data: Endpoint[] }>>();
const mockRefetchModelDefinitions = jest.fn<() => Promise<{ data: ModelDefinition[] }>>();

const mockEndpoints: Endpoint[] = [
  {
    endpoint_id: 'ep-1',
    name: 'endpoint-1',
    model_mappings: [
      {
        mapping_id: 'map-1',
        endpoint_id: 'ep-1',
        model_definition_id: 'md-1',
        model_definition: {
          model_definition_id: 'md-1',
          name: 'model-def-1',
          secret_id: 'secret-1',
          secret_name: 'key-1',
          provider: 'openai',
          model_name: 'gpt-4',
          created_at: 1000,
          last_updated_at: 1000,
          endpoint_count: 1,
        },
        weight: 100,
        created_at: 1000,
      },
    ],
    created_at: 1000,
    last_updated_at: 1000,
  },
];

const mockBindings: EndpointBinding[] = [
  { endpoint_id: 'ep-1', resource_type: 'scorer', resource_id: 'job-1', created_at: 1000, display_name: 'Test Scorer' },
];

const mockModelDefinitions: ModelDefinition[] = [
  {
    model_definition_id: 'md-1',
    name: 'model-def-1',
    secret_id: 'secret-1',
    secret_name: 'key-1',
    provider: 'openai',
    model_name: 'gpt-4',
    created_at: 1000,
    last_updated_at: 1000,
    endpoint_count: 1,
  },
];

jest.mock('./useSecretsQuery', () => ({
  useSecretsQuery: () => ({
    data: [],
    refetch: mockRefetchSecrets,
  }),
}));

jest.mock('./useEndpointsQuery', () => ({
  useEndpointsQuery: () => ({
    data: mockEndpoints,
    refetch: mockRefetchEndpoints,
  }),
}));

jest.mock('./useBindingsQuery', () => ({
  useBindingsQuery: () => ({
    data: mockBindings,
  }),
}));

jest.mock('./useModelDefinitionsQuery', () => ({
  useModelDefinitionsQuery: () => ({
    data: mockModelDefinitions,
    refetch: mockRefetchModelDefinitions,
  }),
}));

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

const mockSecret: SecretInfo = {
  secret_id: 'secret-1',
  secret_name: 'test-key',
  masked_values: { api_key: 'sk-...xyz' },
  provider: 'openai',
  created_at: 1000,
  last_updated_at: 1000,
};

describe('useApiKeysPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockRefetchSecrets.mockResolvedValue({ data: { secrets: [] } });
    mockRefetchEndpoints.mockResolvedValue({ data: mockEndpoints });
    mockRefetchModelDefinitions.mockResolvedValue({ data: mockModelDefinitions });
  });

  afterEach(() => {
    cleanup();
  });

  test('initial state has all modals/drawers closed', () => {
    const { result } = renderHook(() => useApiKeysPage(), { wrapper: createWrapper() });

    expect(result.current.isCreateModalOpen).toBe(false);
    expect(result.current.isDetailsDrawerOpen).toBe(false);
    expect(result.current.isEndpointsDrawerOpen).toBe(false);
    expect(result.current.isBindingsDrawerOpen).toBe(false);
  });

  test('handleCreateClick opens create modal', () => {
    const { result } = renderHook(() => useApiKeysPage(), { wrapper: createWrapper() });

    act(() => {
      result.current.handleCreateClick();
    });

    expect(result.current.isCreateModalOpen).toBe(true);
  });

  test('handleCreateModalClose closes create modal', () => {
    const { result } = renderHook(() => useApiKeysPage(), { wrapper: createWrapper() });

    act(() => {
      result.current.handleCreateClick();
    });
    expect(result.current.isCreateModalOpen).toBe(true);

    act(() => {
      result.current.handleCreateModalClose();
    });
    expect(result.current.isCreateModalOpen).toBe(false);
  });

  test('handleCreateSuccess refetches secrets', () => {
    const { result } = renderHook(() => useApiKeysPage(), { wrapper: createWrapper() });

    act(() => {
      result.current.handleCreateSuccess();
    });

    expect(mockRefetchSecrets).toHaveBeenCalled();
  });

  test('handleKeyClick opens details drawer with selected secret', () => {
    const { result } = renderHook(() => useApiKeysPage(), { wrapper: createWrapper() });

    act(() => {
      result.current.handleKeyClick(mockSecret);
    });

    expect(result.current.isDetailsDrawerOpen).toBe(true);
    expect(result.current.selectedSecret).toEqual(mockSecret);
  });

  test('handleDrawerClose closes details drawer', () => {
    const { result } = renderHook(() => useApiKeysPage(), { wrapper: createWrapper() });

    act(() => {
      result.current.handleKeyClick(mockSecret);
    });
    expect(result.current.isDetailsDrawerOpen).toBe(true);

    act(() => {
      result.current.handleDrawerClose();
    });
    expect(result.current.isDetailsDrawerOpen).toBe(false);
    expect(result.current.selectedSecret).toBeNull();
  });

  test('handleEditSuccess refetches and updates selectedSecret', async () => {
    const updatedSecret = { ...mockSecret, secret_name: 'updated-key', last_updated_at: 2000 };
    mockRefetchSecrets.mockResolvedValue({ data: { secrets: [updatedSecret] } });

    const { result } = renderHook(() => useApiKeysPage(), { wrapper: createWrapper() });

    // Open drawer first so selectedSecret is set
    act(() => {
      result.current.handleKeyClick(mockSecret);
    });
    expect(result.current.selectedSecret).toEqual(mockSecret);

    // Call handleEditSuccess which refetches and updates selectedSecret
    await act(async () => {
      await result.current.handleEditSuccess();
    });

    expect(mockRefetchSecrets).toHaveBeenCalled();
    expect(result.current.selectedSecret).toEqual(updatedSecret);
  });

  test('handleDeleteSuccess refetches all data', async () => {
    const { result } = renderHook(() => useApiKeysPage(), { wrapper: createWrapper() });

    await act(async () => {
      await result.current.handleDeleteSuccess();
    });

    expect(mockRefetchSecrets).toHaveBeenCalled();
    expect(mockRefetchEndpoints).toHaveBeenCalled();
    expect(mockRefetchModelDefinitions).toHaveBeenCalled();
  });

  test('handleEndpointsClick opens endpoints drawer', () => {
    const { result } = renderHook(() => useApiKeysPage(), { wrapper: createWrapper() });

    act(() => {
      result.current.handleEndpointsClick(mockSecret, mockEndpoints);
    });

    expect(result.current.isEndpointsDrawerOpen).toBe(true);
    expect(result.current.endpointsDrawerData).toEqual({
      secret: mockSecret,
      endpoints: mockEndpoints,
    });
  });

  test('handleEndpointsDrawerClose closes endpoints drawer', () => {
    const { result } = renderHook(() => useApiKeysPage(), { wrapper: createWrapper() });

    act(() => {
      result.current.handleEndpointsClick(mockSecret, mockEndpoints);
    });
    expect(result.current.isEndpointsDrawerOpen).toBe(true);

    act(() => {
      result.current.handleEndpointsDrawerClose();
    });
    expect(result.current.isEndpointsDrawerOpen).toBe(false);
  });

  test('handleBindingsClick opens bindings drawer', () => {
    const { result } = renderHook(() => useApiKeysPage(), { wrapper: createWrapper() });

    act(() => {
      result.current.handleBindingsClick(mockSecret, mockBindings);
    });

    expect(result.current.isBindingsDrawerOpen).toBe(true);
    expect(result.current.bindingsDrawerData).toEqual({
      secret: mockSecret,
      bindings: mockBindings,
    });
  });

  test('handleBindingsDrawerClose closes bindings drawer', () => {
    const { result } = renderHook(() => useApiKeysPage(), { wrapper: createWrapper() });

    act(() => {
      result.current.handleBindingsClick(mockSecret, mockBindings);
    });
    expect(result.current.isBindingsDrawerOpen).toBe(true);

    act(() => {
      result.current.handleBindingsDrawerClose();
    });
    expect(result.current.isBindingsDrawerOpen).toBe(false);
  });

  test('exposes allEndpoints from query', () => {
    const { result } = renderHook(() => useApiKeysPage(), { wrapper: createWrapper() });

    expect(result.current.allEndpoints).toEqual(mockEndpoints);
  });
});
