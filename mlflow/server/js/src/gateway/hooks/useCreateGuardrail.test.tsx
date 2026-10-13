import { describe, afterEach, test, jest, expect, beforeEach } from '@jest/globals';
import { renderHook, cleanup, waitFor, act } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { useCreateGuardrail } from './useCreateGuardrail';
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

describe('useCreateGuardrail', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  test('creates guardrail successfully', async () => {
    const mockResponse = {
      guardrail: {
        guardrail_id: 'g-abc123',
        name: 'Safety',
        stage: 'BEFORE',
        action: 'VALIDATION',
        created_at: 1700000000000,
        last_updated_at: 1700000000000,
      },
    };

    jest.spyOn(GatewayApi, 'createGuardrail').mockResolvedValue(mockResponse as any);

    const { result } = renderHook(() => useCreateGuardrail(), { wrapper: createWrapper() });

    await act(async () => {
      await result.current.mutateAsync({
        name: 'Safety',
        scorer_id: 'Safety',
        scorer_version: 1,
        stage: 'BEFORE',
        action: 'VALIDATION',
      });
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(GatewayApi.createGuardrail).toHaveBeenCalledWith({
      name: 'Safety',
      scorer_id: 'Safety',
      scorer_version: 1,
      stage: 'BEFORE',
      action: 'VALIDATION',
    });
  });

  test('handles error state', async () => {
    const mockError = new Error('Scorer not found');
    jest.spyOn(GatewayApi, 'createGuardrail').mockRejectedValue(mockError);

    const { result } = renderHook(() => useCreateGuardrail(), { wrapper: createWrapper() });

    let caughtError: Error | undefined;
    await act(async () => {
      try {
        await result.current.mutateAsync({
          name: 'invalid',
          scorer_id: 'nonexistent',
          scorer_version: 1,
          stage: 'BEFORE',
          action: 'VALIDATION',
        });
      } catch (e) {
        caughtError = e as Error;
      }
    });

    expect(caughtError?.message).toBe('Scorer not found');
    expect(GatewayApi.createGuardrail).toHaveBeenCalled();
  });
});
