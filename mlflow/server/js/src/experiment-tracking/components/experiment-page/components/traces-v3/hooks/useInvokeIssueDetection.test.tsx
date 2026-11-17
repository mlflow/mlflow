import { describe, expect, jest, test } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { fetchAPI } from '../../../../../../common/utils/FetchUtils';
import { useInvokeIssueDetection } from './useInvokeIssueDetection';

jest.mock('../../../../../../common/utils/FetchUtils', () => ({
  fetchAPI: jest.fn(),
  getAjaxUrl: (path: string) => path,
}));
const params = {
  experimentId: '1',
  traceIds: ['trace-1'],
  categories: ['correctness'],
  provider: 'test',
  model: 'synthetic',
};
const response = { job_id: 'job-1', run_id: 'run-1' };
const render = () => {
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  return renderHook(() => ({ first: useInvokeIssueDetection('1'), second: useInvokeIssueDetection('1') }), {
    wrapper: ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    ),
  });
};

describe('issue detection launch concurrency', () => {
  test('equivalent concurrent requests share work and pending state across buttons', async () => {
    let release!: (value: typeof response) => void;
    jest
      .mocked(fetchAPI)
      .mockReset()
      .mockReturnValue(
        new Promise((resolve) => {
          release = resolve;
        }),
      );
    const { result } = render();
    let first!: Promise<unknown>, second!: Promise<unknown>;
    act(() => {
      first = result.current.first.mutateAsync(params);
      second = result.current.second.mutateAsync({ ...params });
    });
    await waitFor(() => expect(fetchAPI).toHaveBeenCalledTimes(1));
    await waitFor(() => {
      expect(result.current.first.isLoading).toBe(true);
      expect(result.current.second.isLoading).toBe(true);
    });
    await act(async () => {
      release(response);
      await Promise.all([first, second]);
    });
    await waitFor(() => {
      expect(result.current.first.isLoading).toBe(false);
      expect(result.current.second.isLoading).toBe(false);
    });
  });

  test('distinct selected scopes remain distinct requests', async () => {
    jest.mocked(fetchAPI).mockReset().mockResolvedValue(response);
    const { result } = render();
    await act(async () => {
      await Promise.all([
        result.current.first.mutateAsync(params),
        result.current.second.mutateAsync({ ...params, traceIds: ['trace-2'] }),
      ]);
    });
    expect(fetchAPI).toHaveBeenCalledTimes(2);
  });

  test('a failure releases the guard for retry', async () => {
    jest.mocked(fetchAPI).mockReset().mockRejectedValueOnce(new Error('offline')).mockResolvedValue(response);
    const { result } = render();
    await act(async () => {
      await expect(result.current.first.mutateAsync(params)).rejects.toThrow('offline');
    });
    await act(async () => {
      expect(await result.current.second.mutateAsync(params)).toEqual(response);
    });
    expect(fetchAPI).toHaveBeenCalledTimes(2);
  });
});
