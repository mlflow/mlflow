import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '../../../../common/utils/reactQueryHooks';
import { useGenAIOverviewAnalyzeState } from './useGenAIOverviewAnalyzeState';

const mockFetchAPI = jest.fn<() => Promise<unknown>>();
const mockRefetchLatestCompletedRun = jest.fn<() => Promise<unknown>>();
const mockUseServerInfo = jest.fn<() => { data?: { store_type: string }; isLoading: boolean }>();

jest.mock('../../../../common/utils/FetchUtils', () => ({
  fetchAPI: () => mockFetchAPI(),
  getAjaxUrl: (path: string) => path,
}));

jest.mock('../../../../workspaces/utils/WorkspaceUtils', () => ({
  useActiveWorkspace: () => 'workspace',
}));

jest.mock('../../../components/experiment-page/hooks/useExperimentEvaluationRunsData', () => ({
  useExperimentEvaluationRunsData: () => ({
    data: [],
    error: null,
    isLoading: false,
    refetch: mockRefetchLatestCompletedRun,
  }),
}));

jest.mock('../../../hooks/useMonitoringConfig', () => ({
  useMonitoringConfig: () => ({ dateNow: new Date('2026-10-21T00:00:00Z') }),
}));

jest.mock('../../../hooks/useServerInfo', () => ({
  useServerInfo: () => mockUseServerInfo(),
}));

const createWrapper = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('useGenAIOverviewAnalyzeState', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockFetchAPI.mockResolvedValue({ issues: [] });
    mockRefetchLatestCompletedRun.mockResolvedValue({});
  });

  test('does not search issues for a file store', async () => {
    mockUseServerInfo.mockReturnValue({ data: { store_type: 'FileStore' }, isLoading: false });
    const { result } = renderHook(() => useGenAIOverviewAnalyzeState('experiment-1'), {
      wrapper: createWrapper(),
    });

    await act(async () => result.current.refetch());

    expect(result.current.state).toEqual({ status: 'never-run' });
    expect(mockRefetchLatestCompletedRun).toHaveBeenCalledTimes(1);
    expect(mockFetchAPI).not.toHaveBeenCalled();
  });

  test('keeps the refetch callback stable', async () => {
    mockUseServerInfo.mockReturnValue({ data: { store_type: 'SqlStore' }, isLoading: false });
    const { result, rerender } = renderHook(() => useGenAIOverviewAnalyzeState('experiment-1'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(mockFetchAPI).toHaveBeenCalledTimes(1));
    const refetch = result.current.refetch;
    rerender();

    expect(result.current.refetch).toBe(refetch);
  });
});
