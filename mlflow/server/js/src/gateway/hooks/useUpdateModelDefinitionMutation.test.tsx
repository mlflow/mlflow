import { describe, afterEach, test, jest, expect, beforeEach } from '@jest/globals';
import { renderHook, cleanup, waitFor, act } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useUpdateModelDefinitionMutation } from './useUpdateModelDefinitionMutation';
import { GatewayApi } from '../api';

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

describe('useUpdateModelDefinitionMutation', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  test('updates model definition successfully', async () => {
    const mockResponse = {
      model_definition: {
        model_definition_id: 'md-123',
        name: 'updated-model-def',
        provider: 'anthropic',
        model_name: 'claude-3',
        secret_id: 'secret-456',
        secret_name: 'my-secret',
        created_at: 1700000000000,
        last_updated_at: 1700000000001,
        endpoint_count: 1,
      },
    };

    jest.spyOn(GatewayApi, 'updateModelDefinition').mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useUpdateModelDefinitionMutation(), { wrapper: createWrapper() });

    await act(async () => {
      await result.current.mutateAsync({
        modelDefinitionId: 'md-123',
        secretId: 'secret-456',
        provider: 'anthropic',
        modelName: 'claude-3',
      });
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(GatewayApi.updateModelDefinition).toHaveBeenCalledWith({
      model_definition_id: 'md-123',
      secret_id: 'secret-456',
      provider: 'anthropic',
      model_name: 'claude-3',
    });
  });

  test('updates model definition with partial data', async () => {
    const mockResponse = {
      model_definition: {
        model_definition_id: 'md-123',
        name: 'model-def',
        provider: 'openai',
        model_name: 'gpt-4',
        secret_id: 'secret-789',
        secret_name: 'new-secret',
        created_at: 1700000000000,
        last_updated_at: 1700000000001,
        endpoint_count: 1,
      },
    };

    jest.spyOn(GatewayApi, 'updateModelDefinition').mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useUpdateModelDefinitionMutation(), { wrapper: createWrapper() });

    await act(async () => {
      await result.current.mutateAsync({
        modelDefinitionId: 'md-123',
        secretId: 'secret-789',
      });
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(GatewayApi.updateModelDefinition).toHaveBeenCalledWith({
      model_definition_id: 'md-123',
      secret_id: 'secret-789',
      provider: undefined,
      model_name: undefined,
    });
  });

  test('handles error state', async () => {
    const mockError = new Error('Model definition not found');
    jest.spyOn(GatewayApi, 'updateModelDefinition').mockRejectedValue(mockError);

    const { result } = renderHook(() => useUpdateModelDefinitionMutation(), { wrapper: createWrapper() });

    let caughtError: Error | undefined;
    await act(async () => {
      try {
        await result.current.mutateAsync({
          modelDefinitionId: 'md-invalid',
        });
      } catch (e) {
        caughtError = e as Error;
      }
    });

    expect(caughtError?.message).toBe('Model definition not found');
    expect(GatewayApi.updateModelDefinition).toHaveBeenCalled();
  });
});
