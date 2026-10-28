import { jest, describe, test, expect, beforeEach } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import React from 'react';

import { QueryClient, QueryClientProvider } from '../../query-client/queryClient';

import { useGenAiExperimentRunsForComparison } from './useGenAiExperimentRunsForComparison';
import { makeRequest } from '../utils/FetchUtils';

jest.mock('../utils/FetchUtils', () => ({
  getAjaxUrl: jest.fn((path: string) => `http://localhost/api/${path}`),
  makeRequest: jest.fn(),
}));

describe('useGenAiExperimentRunsForComparison', () => {
  const mockMakeRequest = jest.mocked(makeRequest);

  beforeEach(() => {
    jest.clearAllMocks();
  });

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

  test('should exclude issue-detection runs from the results', async () => {
    mockMakeRequest.mockResolvedValue({
      runs: [
        {
          info: { run_uuid: 'run-1', run_name: 'Regular Run', start_time: 1625247600000 },
          data: { tags: [] },
        },
        {
          info: { run_uuid: 'run-2', run_name: 'Demo Issue Detection', start_time: 1625247700000 },
          data: { tags: [{ key: 'mlflow.runType', value: 'issue_detection' }] },
        },
        {
          info: { run_uuid: 'run-3', run_name: 'Another Regular Run', start_time: 1625247800000 },
          data: { tags: [{ key: 'mlflow.user', value: 'test@example.com' }] },
        },
      ],
    });

    const { result } = renderHook(() => useGenAiExperimentRunsForComparison('exp-1'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });
    expect(result.current.runInfos?.map((r) => r.runUuid)).toEqual(['run-1', 'run-3']);
  });
});
