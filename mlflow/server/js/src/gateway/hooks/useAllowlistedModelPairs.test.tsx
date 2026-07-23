import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import { renderHook } from '@testing-library/react';
import { useAllowlistedModelPairs } from './useAllowlistedModelPairs';
import { useSecretsQuery } from './useSecretsQuery';

jest.mock('./useSecretsQuery');

const mockSecrets = (secrets: any[], isLoading = false) =>
  jest.mocked(useSecretsQuery).mockReturnValue({
    data: secrets,
    isLoading,
    error: undefined,
    refetch: jest.fn(),
  } as any);

describe('useAllowlistedModelPairs', () => {
  beforeEach(() => {
    jest.clearAllMocks();
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

  test('ignores secrets without allowlisted models', () => {
    mockSecrets([{ secret_id: 'secret-1', secret_name: 'No Models', provider: 'openai' }]);
    const { result } = renderHook(() => useAllowlistedModelPairs());
    expect(result.current.pairs).toEqual([]);
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

  test('surfaces the loading state', () => {
    mockSecrets([], true);
    const { result } = renderHook(() => useAllowlistedModelPairs());
    expect(result.current.isLoading).toBe(true);
  });
});
