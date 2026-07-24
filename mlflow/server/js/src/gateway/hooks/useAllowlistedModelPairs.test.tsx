import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import { renderHook } from '@testing-library/react';
import { useAllowlistedModelPairs } from './useAllowlistedModelPairs';
import { useSecretsQuery } from './useSecretsQuery';
import { useQueries } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

jest.mock('./useSecretsQuery');
jest.mock('@mlflow/mlflow/src/common/utils/reactQueryHooks', () => ({
  ...jest.requireActual<any>('@mlflow/mlflow/src/common/utils/reactQueryHooks'),
  useQueries: jest.fn(),
}));

const mockSecrets = (secrets: any[], isLoading = false) =>
  jest.mocked(useSecretsQuery).mockReturnValue({
    data: secrets,
    isLoading,
    error: undefined,
    refetch: jest.fn(),
  } as any);

// Mock useQueries to return one result per query, in order, each already resolved.
const mockModelQueries = (results: Array<{ models: any[] } | undefined>, isLoading = false) =>
  jest.mocked(useQueries).mockReturnValue(results.map((data) => ({ data, isLoading })) as any);

describe('useAllowlistedModelPairs', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Default: no provider catalogs needed.
    mockModelQueries([]);
  });

  test('returns an empty list when there are no secrets', () => {
    mockSecrets([]);
    const { result } = renderHook(() => useAllowlistedModelPairs());
    expect(result.current.pairs).toEqual([]);
  });

  test('flattens allowlisted models across secrets into stable-sorted pairs', () => {
    mockSecrets([
      {
        secret_id: 'secret-openai',
        secret_name: 'OpenAI Key',
        provider: 'openai',
        allowlisted_models: [{ model: 'gpt-5', provider: 'openai' }],
      },
      {
        secret_id: 'secret-anthropic',
        secret_name: 'Anthropic Key',
        provider: 'anthropic',
        allowlisted_models: [
          { model: 'claude-sonnet', provider: 'anthropic' },
          { model: 'claude-opus', provider: 'anthropic' },
        ],
      },
    ]);

    const { result } = renderHook(() => useAllowlistedModelPairs());

    // Sorted by "Provider · Model" label.
    expect(result.current.pairs.map((p) => p.label)).toEqual([
      'Anthropic · claude-opus',
      'Anthropic · claude-sonnet',
      'OpenAI · gpt-5',
    ]);
    expect(result.current.pairs[0]).toMatchObject({
      secretId: 'secret-anthropic',
      provider: 'anthropic',
      model: 'claude-opus',
      secretName: 'Anthropic Key',
    });
  });

  test('expands a secret with an empty allowlist to all of the provider models', () => {
    mockSecrets([
      {
        secret_id: 'secret-openai',
        secret_name: 'OpenAI Key',
        provider: 'openai',
        allowlisted_models: [],
      },
    ]);
    // One query for the single distinct provider ("openai") that needs expanding.
    mockModelQueries([
      {
        models: [
          { model: 'gpt-5', provider: 'openai' },
          { model: 'gpt-4o', provider: 'openai' },
        ],
      },
    ]);

    const { result } = renderHook(() => useAllowlistedModelPairs());

    expect(result.current.pairs.map((p) => p.label)).toEqual(['OpenAI · gpt-4o', 'OpenAI · gpt-5']);
    expect(result.current.pairs.every((p) => p.secretId === 'secret-openai')).toBe(true);
  });

  test('mixes explicit allowlists and empty (all-models) secrets', () => {
    mockSecrets([
      {
        secret_id: 'secret-anthropic',
        secret_name: 'Anthropic Key',
        provider: 'anthropic',
        allowlisted_models: [{ model: 'claude-opus', provider: 'anthropic' }],
      },
      {
        secret_id: 'secret-openai',
        secret_name: 'OpenAI Key',
        provider: 'openai',
        // Empty → all OpenAI models.
      },
    ]);
    mockModelQueries([{ models: [{ model: 'gpt-5', provider: 'openai' }] }]);

    const { result } = renderHook(() => useAllowlistedModelPairs());

    expect(result.current.pairs.map((p) => p.label)).toEqual(['Anthropic · claude-opus', 'OpenAI · gpt-5']);
  });

  test('dedupes identical (secret, provider, model) triples', () => {
    mockSecrets([
      {
        secret_id: 'secret-1',
        secret_name: 'Key',
        provider: 'openai',
        allowlisted_models: [
          { model: 'gpt-5', provider: 'openai' },
          { model: 'gpt-5', provider: 'openai' },
        ],
      },
    ]);
    const { result } = renderHook(() => useAllowlistedModelPairs());
    expect(result.current.pairs).toHaveLength(1);
  });

  test('surfaces the loading state for secrets', () => {
    mockSecrets([], true);
    const { result } = renderHook(() => useAllowlistedModelPairs());
    expect(result.current.isLoading).toBe(true);
  });

  test('surfaces the loading state while provider model catalogs are fetching', () => {
    mockSecrets([
      { secret_id: 'secret-openai', secret_name: 'OpenAI Key', provider: 'openai', allowlisted_models: [] },
    ]);
    mockModelQueries([undefined], true);
    const { result } = renderHook(() => useAllowlistedModelPairs());
    expect(result.current.isLoading).toBe(true);
  });
});
