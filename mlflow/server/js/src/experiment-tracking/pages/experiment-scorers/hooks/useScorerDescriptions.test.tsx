import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { listScheduledScorers, type MLflowScorer } from '../api';
import { useScorerDescriptions } from './useScorerDescriptions';

jest.mock('../api');

const mockListScheduledScorers = jest.mocked(listScheduledScorers);

const makeScorer = (scorerName: string, serializedScorer: string): MLflowScorer => ({
  experiment_id: 123,
  scorer_name: scorerName,
  scorer_version: 1,
  serialized_scorer: serializedScorer,
  creation_time: 1234567890,
  scorer_id: `${scorerName}-id`,
});

describe('useScorerDescriptions', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    jest.clearAllMocks();
  });

  it('returns only string descriptions from valid serialized scorers', async () => {
    mockListScheduledScorers.mockResolvedValue({
      scorers: [
        makeScorer('quality', JSON.stringify({ description: 'Checks response quality.' })),
        makeScorer('numeric-description', JSON.stringify({ description: 42 })),
        makeScorer('missing-description', JSON.stringify({ name: 'missing-description' })),
        makeScorer('malformed', '{not-json'),
      ],
    });

    const { result } = renderHook(() => useScorerDescriptions('experiment-123'), {
      wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
    });

    await waitFor(() => {
      expect(result.current).toEqual({ quality: 'Checks response quality.' });
    });
    expect(mockListScheduledScorers).toHaveBeenCalledWith('experiment-123');
  });

  it('does not fetch without an experiment id', () => {
    const { result } = renderHook(() => useScorerDescriptions(undefined), {
      wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
    });

    expect(result.current).toEqual({});
    expect(mockListScheduledScorers).not.toHaveBeenCalled();
  });
});
