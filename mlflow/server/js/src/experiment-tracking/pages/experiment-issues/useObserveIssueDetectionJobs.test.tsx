import { expect, jest, test } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { MlflowService } from '../../sdk/MlflowService';
import { MLFLOW_ISSUE_DETECTION_JOB_ID_TAG } from '../../constants';
import { ISSUE_DETECTION_JOB_OBSERVED_EVENT } from '../../components/experiment-page/components/traces-v3/IssueDetectionJobNotifications';
import { useObserveIssueDetectionJobs } from './useObserveIssueDetectionJobs';

jest.mock('../../sdk/MlflowService', () => ({ MlflowService: { searchRuns: jest.fn() } }));
jest.mock('../../../workspaces/utils/WorkspaceUtils', () => ({ useActiveWorkspace: () => 'workspace-a' }));

test('observes every active page once and keeps experiment/workspace identity and the actual start time', async () => {
  const makeRun = (id: string) => ({
    info: { runUuid: id, startTime: 12345 },
    data: { tags: [{ key: MLFLOW_ISSUE_DETECTION_JOB_ID_TAG, value: `job-${id}` }] },
  });
  jest
    .mocked(MlflowService.searchRuns)
    .mockImplementation(
      async ({ page_token }: { page_token?: string }) =>
        (page_token
          ? { runs: [makeRun('second')] }
          : { runs: [makeRun('first'), { info: { runUuid: 'untagged' } }], next_page_token: 'next' }) as any,
    );
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, cacheTime: 0 } } });
  const events: any[] = [];
  const listener = (event: Event) => events.push((event as CustomEvent).detail);
  window.addEventListener(ISSUE_DETECTION_JOB_OBSERVED_EVENT, listener);
  try {
    const { unmount } = renderHook(() => useObserveIssueDetectionJobs('exp-1'), {
      wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>,
    });
    await waitFor(() => expect(events).toHaveLength(2));
    expect(events[1]).toEqual(
      expect.objectContaining({
        experimentId: 'exp-1',
        workspace: 'workspace-a',
        runId: 'second',
        jobId: 'job-second',
        startTimeMs: 12345,
      }),
    );
    expect(MlflowService.searchRuns).toHaveBeenCalledWith(
      expect.objectContaining({
        filter: expect.stringContaining("AND attributes.status = 'RUNNING'"),
        page_token: 'next',
      }),
    );
    await client.invalidateQueries();
    expect(events).toHaveLength(2);
    unmount();
  } finally {
    window.removeEventListener(ISSUE_DETECTION_JOB_OBSERVED_EVENT, listener);
    client.clear();
  }
});
