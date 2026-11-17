import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { rest } from 'msw';

import { renderWithDesignSystem, waitFor } from '../../../common/utils/TestUtils.react18';
import { setupServer } from '../../../common/utils/setup-msw';
import { QueryClient, QueryClientProvider } from '../../../common/utils/reactQueryHooks';
import { GET_ISSUE_QUERY_KEY } from '../../components/run-page/hooks/useGetIssueQuery';
import { SEARCH_ISSUES_QUERY_KEY } from '../../components/run-page/hooks/useSearchIssuesQuery';
import { MLFLOW_ISSUE_DETECTION_JOB_ID_TAG } from '../../constants';
import { useLatestIssueDetection } from './useIssueDetection';
import { ISSUE_EVIDENCE_TRACES_QUERY_KEY } from './useIssueEvidenceTraces';

const server = setupServer(
  // One FINISHED detection run carrying a job id, which is what makes the hook treat it as a
  // completed submission and refresh what it produced.
  rest.post('*/ajax-api/2.0/mlflow/runs/search', (_req, res, ctx) =>
    res(
      ctx.json({
        runs: [
          {
            info: { run_id: 'run-1', status: 'FINISHED', start_time: Date.now() },
            data: { tags: [{ key: MLFLOW_ISSUE_DETECTION_JOB_ID_TAG, value: 'job-1' }] },
          },
        ],
      }),
    ),
  ),
  rest.get('*/ajax-api/3.0/mlflow/jobs/:jobId', (_req, res, ctx) => res(ctx.json({ status: 3 }))),
);

const Probe = () => {
  useLatestIssueDetection('1');
  return null;
};

describe('useLatestIssueDetection invalidation', () => {
  beforeEach(() => {
    server.resetHandlers();
  });

  test('a completed run refreshes the open issue and its evidence, not just the list', async () => {
    // A run that matches a finding to an existing issue grows that issue's affected-trace count
    // while the open detail pane and its evidence stay cached, so the pane showed one trace next
    // to a count of two.
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, cacheTime: 0 } } });
    const invalidated: unknown[] = [];
    jest.spyOn(client, 'invalidateQueries').mockImplementation(((key: unknown) => {
      invalidated.push(key);
      return Promise.resolve();
    }) as any);

    renderWithDesignSystem(
      <QueryClientProvider client={client}>
        <Probe />
      </QueryClientProvider>,
    );

    await waitFor(() => expect(invalidated.flat()).toContain(SEARCH_ISSUES_QUERY_KEY));
    const keys = invalidated.flat();
    expect(keys).toContain(GET_ISSUE_QUERY_KEY);
    expect(keys).toContain(ISSUE_EVIDENCE_TRACES_QUERY_KEY);
  });
});
