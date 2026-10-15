import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { rest } from 'msw';
import userEvent from '@testing-library/user-event';
import { act, fireEvent, renderWithDesignSystem, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { setupServer } from '../../../common/utils/setup-msw';
import { shouldEnableExperimentIssuesTab } from '../../../common/utils/FeatureUtils';
import { MemoryRouter, Route, Routes, useNavigate } from '../../../common/utils/RoutingUtils';
import { QueryClient, QueryClientProvider } from '../../../common/utils/reactQueryHooks';
import { ServerInfoProvider } from '../../hooks/useServerInfo';
import type { Issue } from '../../components/run-page/hooks/useSearchIssuesQuery';
import { ISSUE_DETECTION_JOB_SUBMITTED_EVENT } from '../../components/experiment-page/components/traces-v3/IssueDetectionJobNotifications';
import { useExperimentPermission } from '../../hooks/useExperimentPermission';
import ExperimentIssuesPage from './ExperimentIssuesPage';

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableExperimentIssuesTab: jest.fn(),
}));

jest.mock('../../hooks/useExperimentPermission', () => ({ useExperimentPermission: jest.fn(() => true) }));
jest.mock('../../components/run-page/IssueTracesPanel', () => ({
  IssueTracesPanel: ({ issue }: { issue: Issue }) => <div>References for {issue.issue_id}</div>,
}));
jest.mock('../../components/experiment-page/components/traces-v3/IssueDetectionModal', () => ({
  IssueDetectionModal: ({ availableTraceIds }: { availableTraceIds: string[] }) => (
    <div role="dialog">Selected traces: {availableTraceIds.join(', ')}</div>
  ),
}));
const makeIssue = (id: string, name = id): Issue => ({
  issue_id: id,
  experiment_id: '1',
  name,
  description: `Description for ${id}`,
  status: 'pending',
  severity: 'high',
  created_timestamp: 100,
  last_updated_timestamp: 200,
  categories: ['latency'],
  trace_count: 2,
});
let issues: Issue[];
let searches: Record<string, unknown>[];
let latestRun:
  | { info: { run_id: string; status: string }; data: { tags: { key: string; value: string }[] } }
  | undefined;
let jobStatus: string;
const server = setupServer(
  rest.get('*/ajax-api/3.0/mlflow/server-info', (_req, res, ctx) =>
    res(ctx.json({ store_type: 'SqlStore', workspaces_enabled: false, features_enabled: { gateway: true } })),
  ),
  rest.post('*/ajax-api/3.0/mlflow/issues/search', async (req, res, ctx) => {
    const body = await req.json();
    searches.push(body);
    const matches = issues.filter(
      (issue) =>
        (!body.name || issue.name.toLowerCase().includes(body.name.toLowerCase())) &&
        (!body.statuses?.length || body.statuses.includes(issue.status)),
    );
    const offset = Number(body.page_token ?? 0);
    return res(
      ctx.json({
        issues: matches.slice(offset, offset + body.max_results),
        total_count: matches.length,
        next_page_token: offset + body.max_results < matches.length ? String(offset + body.max_results) : undefined,
      }),
    );
  }),
  rest.get('*/ajax-api/3.0/mlflow/issues/:id', (req, res, ctx) => {
    const issue = issues.find((item) => item.issue_id === req.params['id']);
    return issue ? res(ctx.json({ issue })) : res(ctx.status(404), ctx.json({ message: 'Missing Issue' }));
  }),
  rest.patch('*/ajax-api/3.0/mlflow/issues/:id', async (req, res, ctx) => {
    const body = await req.json();
    const issue = issues.find((item) => item.issue_id === req.params['id']);
    if (issue) Object.assign(issue, { status: body.status });
    return res(ctx.json({ issue }));
  }),
  rest.post('*/ajax-api/2.0/mlflow/runs/search', (_req, res, ctx) =>
    res(ctx.json({ runs: latestRun ? [latestRun] : [] })),
  ),
  rest.get('*/ajax-api/3.0/mlflow/jobs/job-1', (_req, res, ctx) =>
    res(ctx.json({ status: jobStatus, result: { issues: issues.length, total_traces_analyzed: 50 } })),
  ),
  rest.post('*/ajax-api/3.0/mlflow/traces/search', (_req, res, ctx) =>
    res(ctx.json({ traces: [{ trace_id: 'trace-1' }, { trace_id: 'trace-2' }] })),
  ),
);
const Navigation = () => {
  const navigate = useNavigate();
  return <button onClick={() => navigate(-1)}>Back</button>;
};
const renderPage = (query = '') => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, cacheTime: 0 } } });
  const result = renderWithDesignSystem(
    <QueryClientProvider client={client}>
      <ServerInfoProvider>
        <MemoryRouter initialEntries={[`/experiments/1/issues${query}`]}>
          <Navigation />
          <Routes>
            <Route path="/experiments/:experimentId/issues" element={<ExperimentIssuesPage />} />
          </Routes>
        </MemoryRouter>
      </ServerInfoProvider>
    </QueryClientProvider>,
  );
  return { ...result, client };
};
beforeEach(() => {
  jest.mocked(shouldEnableExperimentIssuesTab).mockReturnValue(true);
  jest.mocked(useExperimentPermission).mockReturnValue(true);
  issues = [makeIssue('iss-a', 'Slow response'), makeIssue('iss-b', 'Wrong answer')];
  issues[0].root_causes = ['Repeated model calls'];
  searches = [];
  latestRun = undefined;
  jobStatus = 'SUCCEEDED';
});
describe('Experiment Issues', () => {
  test.each(['', '?selectedIssueId=iss-a'])(
    'does not load Issues or detection data from a direct link when the flag is disabled (%s)',
    async (query) => {
      jest.mocked(shouldEnableExperimentIssuesTab).mockReturnValue(false);
      const fetchIssueData = jest.fn();
      server.use(
        rest.all(/\/ajax-api\/(?:2|3)\.0\/mlflow\/(?:issues|runs|jobs|traces)\//, (_req, res, ctx) => {
          fetchIssueData();
          return res(ctx.json({}));
        }),
      );
      renderPage(query);
      expect(await screen.findByRole('heading', { name: 'Page Not Found' })).toBeInTheDocument();
      expect(screen.queryByRole('button', { name: 'Detect issues' })).not.toBeInTheDocument();
      expect(screen.queryByRole('textbox', { name: 'Search issues' })).not.toBeInTheDocument();
      expect(fetchIssueData).not.toHaveBeenCalled();
    },
  );
  test('reads SQL issues, renders legacy causes and preserves selection through back navigation', async () => {
    renderPage('?selectedIssueId=iss-a');
    expect(await screen.findByText('Repeated model calls')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /Wrong answer/ }));
    expect(await screen.findByText('References for iss-b')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Back' }));
    expect(await screen.findByText('References for iss-a')).toBeInTheDocument();
    expect(searches).toContainEqual(
      expect.objectContaining({
        experiment_id: '1',
        view: 'basic',
        include_total_count: true,
        include_trace_count: true,
      }),
    );
  });
  test('searches beyond the loaded page and loads the next page from SQL', async () => {
    issues = Array.from({ length: 51 }, (_, index) =>
      makeIssue(`iss-${index}`, index === 50 ? 'Later match' : `Issue ${index}`),
    );
    renderPage();
    await userEvent.click(await screen.findByRole('button', { name: 'Load more' }));
    expect(await screen.findByRole('button', { name: /Later match/ })).toBeInTheDocument();
    expect(searches).toContainEqual(expect.objectContaining({ page_token: '50' }));
    fireEvent.change(screen.getByRole('textbox', { name: 'Search issues' }), { target: { value: 'Later match' } });
    await waitFor(() =>
      expect(searches).toContainEqual(expect.objectContaining({ name: 'Later match', max_results: 50 })),
    );
    await waitFor(() => expect(screen.queryByRole('button', { name: /Issue 0/ })).not.toBeInTheDocument());
    expect(screen.getByRole('button', { name: /Later match/ })).toBeInTheDocument();
  });
  test('keeps loaded rows when loading another page fails', async () => {
    server.use(
      rest.post('*/ajax-api/3.0/mlflow/issues/search', async (req, res, ctx) => {
        const body = await req.json();
        return body.page_token
          ? res(ctx.status(500))
          : res(ctx.json({ issues: issues.slice(0, 1), next_page_token: 'next' }));
      }),
    );
    renderPage();
    await userEvent.click(await screen.findByRole('button', { name: 'Load more' }));
    expect(await screen.findByText('Unable to refresh or load more issues')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Slow response/ })).toBeInTheDocument();
  });
  test('rolls back optimistic triage after a failed patch', async () => {
    server.use(
      rest.patch('*/ajax-api/3.0/mlflow/issues/:id', (_req, res, ctx) =>
        res(ctx.delay(200), ctx.status(403), ctx.json({ message: 'Read only' })),
      ),
    );
    renderPage('?selectedIssueId=iss-a');
    await userEvent.click(await screen.findByRole('button', { name: 'Resolve' }));
    expect(await screen.findByRole('button', { name: 'Reopen' })).toBeInTheDocument();
    expect(await screen.findByText('Unable to update issue')).toBeInTheDocument();
    expect(await screen.findByRole('button', { name: 'Resolve' })).toBeInTheDocument();
    expect(issues[0].status).toBe('pending');
  });
  test('resolves and reopens a linked issue even when its row is filtered out', async () => {
    renderPage('?selectedIssueId=iss-a');
    await userEvent.click(await screen.findByRole('button', { name: 'Resolve' }));
    await waitFor(() => expect(issues[0].status).toBe('resolved'));
    await userEvent.click(await screen.findByRole('button', { name: 'Reopen' }));
    await waitFor(() => expect(issues[0].status).toBe('pending'));
    expect(await screen.findByText('References for iss-a')).toBeInTheDocument();
  });
  test('opens the existing detector with recent trace IDs', async () => {
    renderPage();
    await userEvent.click(await screen.findByRole('button', { name: 'Detect issues' }));
    expect(await screen.findByRole('dialog')).toHaveTextContent('Selected traces: trace-1, trace-2');
  });
  test('refreshes SQL issues when detection finishes and recovers progress after reload', async () => {
    issues = [];
    latestRun = {
      info: { run_id: 'run-1', status: 'RUNNING' },
      data: { tags: [{ key: 'mlflow.issueDetection.jobId', value: 'job-1' }] },
    };
    jobStatus = 'RUNNING';
    const { client } = renderPage();
    expect(await screen.findByText('Detecting issues')).toBeInTheDocument();
    issues = [makeIssue('iss-detected', 'Detected by job')];
    jobStatus = 'SUCCEEDED';
    latestRun.info.status = 'FINISHED';
    await act(async () => {
      await client.invalidateQueries(['FETCH_JOB_STATUS', 'job-1']);
    });
    expect(await screen.findByRole('button', { name: /Detected by job/ })).toBeInTheDocument();
    expect(await screen.findByText('Issue detection completed')).toBeInTheDocument();
  });
  test('discovers a newly submitted job without navigating away', async () => {
    renderPage();
    await screen.findByText('References for iss-a');
    latestRun = {
      info: { run_id: 'run-1', status: 'RUNNING' },
      data: { tags: [{ key: 'mlflow.issueDetection.jobId', value: 'job-1' }] },
    };
    jobStatus = 'RUNNING';
    act(() => window.dispatchEvent(new CustomEvent(ISSUE_DETECTION_JOB_SUBMITTED_EVENT)));
    expect(await screen.findByText('Detecting issues')).toBeInTheDocument();
  });
  test('distinguishes a missing Issue from denied access', async () => {
    const { unmount } = renderPage('?selectedIssueId=iss-missing');
    expect(await screen.findByText('Issue not found in this experiment')).toBeInTheDocument();
    unmount();
    server.use(rest.get('*/ajax-api/3.0/mlflow/issues/:id', (_req, res, ctx) => res(ctx.status(403))));
    renderPage('?selectedIssueId=iss-a');
    expect(await screen.findByText('You do not have permission to view these issues')).toBeInTheDocument();
  });
  test('disables detection and hides triage and cancel actions for read-only users', async () => {
    jest.mocked(useExperimentPermission).mockReturnValue(false);
    latestRun = {
      info: { run_id: 'run-1', status: 'RUNNING' },
      data: { tags: [{ key: 'mlflow.issueDetection.jobId', value: 'job-1' }] },
    };
    jobStatus = 'RUNNING';
    renderPage();
    await screen.findByText('References for iss-a');
    expect(screen.getByRole('button', { name: 'Detect issues' })).toBeDisabled();
    expect(screen.queryByRole('button', { name: 'Resolve' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Cancel detection' })).not.toBeInTheDocument();
  });
  test('shows no matches separately from a clean scan', async () => {
    renderPage('?issueSearch=no-match');
    expect(await screen.findByText('No issues match these filters.')).toBeInTheDocument();
  });
  test.each(['FINISHED', 'FAILED'])('only reports a clean scan after successful completion (%s)', async (status) => {
    issues = [];
    latestRun = {
      info: { run_id: 'run-1', status },
      data: {
        tags: [
          { key: 'mlflow.issueDetection.result.issues', value: '0' },
          { key: 'mlflow.issueDetection.result.totalTracesAnalyzed', value: '50' },
        ],
      },
    };
    renderPage();
    if (status === 'FINISHED') {
      expect(
        await screen.findByText('The latest detection found no issues in the selected traces.'),
      ).toBeInTheDocument();
    } else {
      expect(await screen.findByText('Issue detection failed')).toBeInTheDocument();
      expect(
        screen.queryByText('The latest detection found no issues in the selected traces.'),
      ).not.toBeInTheDocument();
    }
  });
  test('cancels a recovered detection job after confirmation', async () => {
    latestRun = {
      info: { run_id: 'run-1', status: 'RUNNING' },
      data: { tags: [{ key: 'mlflow.issueDetection.jobId', value: 'job-1' }] },
    };
    jobStatus = 'RUNNING';
    const canceled = jest.fn();
    server.use(
      rest.patch('*/ajax-api/3.0/mlflow/jobs/cancel/job-1', (_req, res, ctx) => {
        canceled();
        jobStatus = 'CANCELED';
        return res(ctx.json({ status: 'CANCELED' }));
      }),
      rest.post('*/ajax-api/2.0/mlflow/runs/update', (_req, res, ctx) => res(ctx.json({}))),
    );
    renderPage();
    await userEvent.click(await screen.findByRole('button', { name: 'Cancel detection' }));
    expect(canceled).not.toHaveBeenCalled();
    const confirmButtons = screen.getAllByRole('button', { name: 'Cancel detection' });
    await userEvent.click(confirmButtons[confirmButtons.length - 1]);
    expect(await screen.findByText('Issue detection canceled')).toBeInTheDocument();
    expect(canceled).toHaveBeenCalledTimes(1);
  });
  test('does not call Issue APIs for unsupported or proxied storage', async () => {
    server.use(rest.get('*/ajax-api/3.0/mlflow/server-info', (_req, res, ctx) => res(ctx.json({ store_type: null }))));
    renderPage();
    expect(await screen.findByText('The Issues tab requires an MLflow server with SQL storage.')).toBeInTheDocument();
    expect(searches).toEqual([]);
  });
});
