import { describe, afterEach, test, jest, expect, beforeEach } from '@jest/globals';
import { renderHook, cleanup, act } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useApiKeysPage } from './useApiKeysPage';
import type { SecretInfo, Endpoint, EndpointBinding, ModelDefinition } from '../types';

const mockRefetchSecrets = jest.fn<() => Promise<{ data: SecretInfo[] }>>();
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
    mockRefetchSecrets.mockResolvedValue({ data: [] });
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
    expect(result.current.isEditModalOpen).toBe(false);
    expect(result.current.isDeleteModalOpen).toBe(false);
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

  test('handleEditClick opens edit modal with secret', () => {
    const { result } = renderHook(() => useApiKeysPage(), { wrapper: createWrapper() });

    act(() => {
      result.current.handleEditClick(mockSecret);
    });

    expect(result.current.isEditModalOpen).toBe(true);
    expect(result.current.editingSecret).toEqual(mockSecret);
  });

  test('handleEditModalClose closes edit modal', () => {
    const { result } = renderHook(() => useApiKeysPage(), { wrapper: createWrapper() });

    act(() => {
      result.current.handleEditClick(mockSecret);
    });
    expect(result.current.isEditModalOpen).toBe(true);

    act(() => {
      result.current.handleEditModalClose();
    });
    expect(result.current.isEditModalOpen).toBe(false);
    expect(result.current.editingSecret).toBeNull();
  });

  test('handleDeleteClick opens delete modal with data', () => {
    const { result } = renderHook(() => useApiKeysPage(), { wrapper: createWrapper() });

    act(() => {
      result.current.handleDeleteClick(mockSecret, mockModelDefinitions, mockEndpoints, 1);
    });

    expect(result.current.isDeleteModalOpen).toBe(true);
    expect(result.current.deleteModalData).toEqual({
      secret: mockSecret,
      modelDefinitions: mockModelDefinitions,
      endpoints: mockEndpoints,
      bindingCount: 1,
    });
  });

  test('handleDeleteFromDrawer computes model definitions, endpoints, and binding count', () => {
    const { result } = renderHook(() => useApiKeysPage(), { wrapper: createWrapper() });

    act(() => {
      result.current.handleDeleteFromDrawer(mockSecret);
    });

    expect(result.current.isDeleteModalOpen).toBe(true);
    expect(result.current.deleteModalData?.secret).toEqual(mockSecret);
    expect(result.current.deleteModalData?.modelDefinitions).toEqual(mockModelDefinitions);
    expect(result.current.deleteModalData?.endpoints).toEqual(mockEndpoints);
    expect(result.current.deleteModalData?.bindingCount).toBe(1);
  });

  test('handleDeleteModalClose closes delete modal', () => {
    const { result } = renderHook(() => useApiKeysPage(), { wrapper: createWrapper() });

    act(() => {
      result.current.handleDeleteClick(mockSecret, [], [], 0);
    });
    expect(result.current.isDeleteModalOpen).toBe(true);

    act(() => {
      result.current.handleDeleteModalClose();
    });
    expect(result.current.isDeleteModalOpen).toBe(false);
    expect(result.current.deleteModalData).toBeNull();
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
