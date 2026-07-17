import { describe, afterEach, test, jest, expect, beforeEach } from '@jest/globals';
import { renderHook, cleanup, waitFor, act } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { useDeleteEndpoint } from './useDeleteEndpoint';
import { GatewayApi } from '../api';
import type { EndpointModelMapping } from '../types';

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

const makeMapping = (modelDefinitionId: string): EndpointModelMapping => ({
  mapping_id: `mapping-${modelDefinitionId}`,
  endpoint_id: 'ep-1',
  model_definition_id: modelDefinitionId,
  weight: 1,
  created_at: 0,
});

describe('useDeleteEndpoint', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  test('deletes endpoint and all associated model definitions', async () => {
    jest.spyOn(GatewayApi, 'deleteEndpoint').mockResolvedValue(undefined);
    jest.spyOn(GatewayApi, 'deleteModelDefinition').mockResolvedValue(undefined);

    const { result } = renderHook(() => useDeleteEndpoint(), { wrapper: createWrapper() });

    await act(async () => {
      await result.current.mutateAsync({
        endpointId: 'ep-1',
        modelMappings: [makeMapping('md-1'), makeMapping('md-2')],
      });
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(GatewayApi.deleteEndpoint).toHaveBeenCalledWith('ep-1');
    expect(GatewayApi.deleteModelDefinition).toHaveBeenCalledWith('md-1');
    expect(GatewayApi.deleteModelDefinition).toHaveBeenCalledWith('md-2');
  });

  test('succeeds even when a model definition deletion fails', async () => {
    jest.spyOn(GatewayApi, 'deleteEndpoint').mockResolvedValue(undefined);
    jest.spyOn(GatewayApi, 'deleteModelDefinition').mockRejectedValue(new Error('still in use'));

    const { result } = renderHook(() => useDeleteEndpoint(), { wrapper: createWrapper() });

    await act(async () => {
      await result.current.mutateAsync({
        endpointId: 'ep-1',
        modelMappings: [makeMapping('md-1')],
      });
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(GatewayApi.deleteEndpoint).toHaveBeenCalledWith('ep-1');
    expect(GatewayApi.deleteModelDefinition).toHaveBeenCalledWith('md-1');
  });

  test('fails and does not delete model definitions when endpoint deletion fails', async () => {
    jest.spyOn(GatewayApi, 'deleteEndpoint').mockRejectedValue(new Error('endpoint not found'));
    jest.spyOn(GatewayApi, 'deleteModelDefinition').mockResolvedValue(undefined);

    const { result } = renderHook(() => useDeleteEndpoint(), { wrapper: createWrapper() });

    let caughtError: Error | undefined;
    await act(async () => {
      try {
        await result.current.mutateAsync({
          endpointId: 'ep-1',
          modelMappings: [makeMapping('md-1')],
        });
      } catch (e) {
        caughtError = e as Error;
      }
    });

    expect(caughtError?.message).toBe('endpoint not found');
    expect(GatewayApi.deleteEndpoint).toHaveBeenCalledWith('ep-1');
    expect(GatewayApi.deleteModelDefinition).not.toHaveBeenCalled();
  });

  test('deletes endpoint with no model mappings', async () => {
    jest.spyOn(GatewayApi, 'deleteEndpoint').mockResolvedValue(undefined);
    jest.spyOn(GatewayApi, 'deleteModelDefinition').mockResolvedValue(undefined);

    const { result } = renderHook(() => useDeleteEndpoint(), { wrapper: createWrapper() });

    await act(async () => {
      await result.current.mutateAsync({ endpointId: 'ep-1', modelMappings: [] });
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(GatewayApi.deleteEndpoint).toHaveBeenCalledWith('ep-1');
    expect(GatewayApi.deleteModelDefinition).not.toHaveBeenCalled();
  });
});
