import { jest, describe, beforeEach, expect, test } from '@jest/globals';
import React from 'react';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { fetchAPI } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { useInvokeIssueDetection } from './useInvokeIssueDetection';
import { LATEST_ISSUE_DETECTION_QUERY_KEY } from '../../../../../pages/experiment-issues/useIssueDetection';

jest.mock('@mlflow/mlflow/src/common/utils/FetchUtils', () => ({
  fetchAPI: jest.fn(),
  getAjaxUrl: (url: string) => url,
}));

const mockFetchAPI = jest.mocked(fetchAPI);

/** Answers DetectIssues with `runId`, then the run lookup with `tags`. */
const givenStartedRun = (runId: string | undefined, tags: { key: string; value: string }[] = []) => {
  mockFetchAPI.mockImplementation(((url: string) =>
    Promise.resolve(
      url.startsWith('ajax-api/3.0/mlflow/issues/detect')
        ? { issue_detection_run: runId ? { run_id: runId, status: 'running' } : undefined }
        : { run: { data: { tags } } },
    )) as unknown as typeof fetchAPI);
};

describe('useInvokeIssueDetection', () => {
  let queryClient: QueryClient;

  const render = () =>
    renderHook(() => useInvokeIssueDetection(), {
      wrapper: ({ children }: { children: React.ReactNode }) => (
        <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
      ),
    });

  beforeEach(() => {
    jest.clearAllMocks();
    queryClient = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  });

  test('starts a direct provider run on the chosen secret through DetectIssues', async () => {
    givenStartedRun('run-1', [{ key: 'mlflow.issueDetection.jobId', value: 'job-1' }]);
    const { result } = render();

    const response = await result.current.mutateAsync({
      experimentId: 'exp-1',
      traceIds: ['tr-1', 'tr-2'],
      provider: 'openai',
      model: 'gpt-5',
      secret_id: 'secret-1',
    });

    expect(response).toEqual({ job_id: 'job-1', run_id: 'run-1' });
    expect(mockFetchAPI).toHaveBeenNthCalledWith(1, 'ajax-api/3.0/mlflow/issues/detect', {
      method: 'POST',
      body: { experiment_id: 'exp-1', trace_ids: ['tr-1', 'tr-2'], model: 'openai:/gpt-5', secret_id: 'secret-1' },
    });
    expect(mockFetchAPI).toHaveBeenNthCalledWith(2, 'ajax-api/2.0/mlflow/runs/get?run_id=run-1');
  });

  test('runs a gateway endpoint without a secret', async () => {
    givenStartedRun('run-1', [{ key: 'mlflow.issueDetection.jobId', value: 'job-1' }]);
    const { result } = render();

    await result.current.mutateAsync({
      experimentId: 'exp-1',
      traceIds: ['tr-1'],
      provider: 'openai',
      model: 'gpt-5',
      secret_id: 'secret-1',
      endpoint_name: 'my-endpoint',
    });

    expect(mockFetchAPI.mock.calls[0][1]).toEqual({
      method: 'POST',
      body: { experiment_id: 'exp-1', trace_ids: ['tr-1'], model: 'gateway:/my-endpoint', secret_id: undefined },
    });
  });

  test('a run with no recorded job still resolves and refreshes the latest run', async () => {
    givenStartedRun('run-1');
    const invalidate = jest.spyOn(queryClient, 'invalidateQueries');
    const { result } = render();

    const response = await result.current.mutateAsync({
      experimentId: 'exp-1',
      traceIds: ['tr-1'],
      provider: 'openai',
      model: 'gpt-5',
    });

    expect(response).toEqual({ job_id: undefined, run_id: 'run-1' });
    await waitFor(() => expect(invalidate).toHaveBeenCalledWith([LATEST_ISSUE_DETECTION_QUERY_KEY]));
  });

  test('fails when the server returns no run', async () => {
    givenStartedRun(undefined);
    const { result } = render();

    await expect(
      result.current.mutateAsync({ experimentId: 'exp-1', traceIds: ['tr-1'], provider: 'openai', model: 'gpt-5' }),
    ).rejects.toThrow('Issue detection started without returning its run.');
    expect(mockFetchAPI).toHaveBeenCalledTimes(1);
  });
});
