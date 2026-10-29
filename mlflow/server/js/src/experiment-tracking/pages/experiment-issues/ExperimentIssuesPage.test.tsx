import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { rest } from 'msw';
import userEvent from '@testing-library/user-event';
import {
  act,
  fireEvent,
  renderWithDesignSystem,
  screen,
  waitFor,
  within,
} from '../../../common/utils/TestUtils.react18';
import { setupServer } from '../../../common/utils/setup-msw';
import { shouldEnableExperimentIssuesTab } from '../../../common/utils/FeatureUtils';
import { MemoryRouter, Route, Routes, useLocation, useNavigate } from '../../../common/utils/RoutingUtils';
import { QueryClient, QueryClientProvider } from '../../../common/utils/reactQueryHooks';
import { ServerInfoProvider } from '../../hooks/useServerInfo';
import type { Issue } from '../../components/run-page/hooks/useSearchIssuesQuery';
import {
  clearSubmittedIssueDetectionJob,
  IssueDetectionJobNotifications,
  recordSubmittedIssueDetectionJob,
} from '../../components/experiment-page/components/traces-v3/IssueDetectionJobNotifications';
import { DEFAULT_DETECT_ISSUES_GUIDANCE_STORAGE_KEY } from '@databricks/web-shared/genai-traces-table/components/DetectIssuesButton';
import ExperimentIssuesPage from './ExperimentIssuesPage';

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableExperimentIssuesTab: jest.fn(),
}));

jest.mock('../../../account/hooks', () => ({
  useIsAuthAvailable: () => true,
  useMyPermissionsQuery: () => ({ data: { is_admin: false, permissions: [] } }),
}));
jest.mock('../../components/experiment-page/components/traces-v3/IssueDetectionModal', () => ({
  IssueDetectionModal: ({ availableTraceIds }: { availableTraceIds: string[] }) => (
    <div role="dialog">
      Selected traces: {availableTraceIds.join(', ')}
      <input aria-label="Detection model draft" />
    </div>
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
let responsePageSize: number | undefined;
let latestRun:
  | { info: { run_id: string; status: string; start_time?: number }; data: { tags: { key: string; value: string }[] } }
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
    const pageSize = responsePageSize ?? body.max_results;
    return res(
      ctx.json({
        issues: matches.slice(offset, offset + pageSize),
        total_count: matches.length,
        next_page_token: offset + pageSize < matches.length ? String(offset + pageSize) : undefined,
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
  const location = useLocation();
  return (
    <>
      <button onClick={() => navigate(-1)}>Back</button>
      <output aria-label="Current URL">{location.pathname + location.search}</output>
    </>
  );
};
const renderPage = (query = '', withNotifications = false) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, cacheTime: 0 } } });
  const result = renderWithDesignSystem(
    <QueryClientProvider client={client}>
      <ServerInfoProvider>
        <MemoryRouter initialEntries={[`/experiments/1/issues${query}`]}>
          <Navigation />
          {withNotifications && <IssueDetectionJobNotifications />}
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
  localStorage.setItem(`${DEFAULT_DETECT_ISSUES_GUIDANCE_STORAGE_KEY}_v1`, JSON.stringify(true));
  clearSubmittedIssueDetectionJob();
  jest.mocked(shouldEnableExperimentIssuesTab).mockReturnValue(true);
  issues = [makeIssue('iss-a', 'Slow response'), makeIssue('iss-b', 'Wrong answer')];
  searches = [];
  responsePageSize = undefined;
  latestRun = undefined;
  jobStatus = 'SUCCEEDED';
});
describe('Experiment Issues', () => {
  test.each(['network', 'server'])('allows retry when server information fails (%s)', async (failure) => {
    server.use(
      rest.get('*/ajax-api/3.0/mlflow/server-info', (_req, res, ctx) =>
        failure === 'network' ? res.networkError('Connection failed') : res(ctx.status(500)),
      ),
    );
    renderPage();
    expect(await screen.findByText('Unable to load server information')).toBeInTheDocument();
    expect(screen.queryByText('The Issues tab requires an MLflow server with SQL storage.')).not.toBeInTheDocument();
    expect(searches).toHaveLength(0);
    server.use(
      rest.get('*/ajax-api/3.0/mlflow/server-info', (_req, res, ctx) => res(ctx.json({ store_type: 'SqlStore' }))),
    );
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(await screen.findByText('Description for iss-a')).toBeInTheDocument();
  });
  test('shows the SQL storage requirement for a known unsupported store', async () => {
    server.use(
      rest.get('*/ajax-api/3.0/mlflow/server-info', (_req, res, ctx) => res(ctx.json({ store_type: 'FileStore' }))),
    );
    renderPage();
    expect(await screen.findByText('The Issues tab requires an MLflow server with SQL storage.')).toBeInTheDocument();
    expect(screen.queryByText('Unable to load server information')).not.toBeInTheDocument();
    expect(searches).toHaveLength(0);
  });
  test('writes the initial selection to the URL and selects the first match of a new title search', async () => {
    renderPage('?issueSeverity=high');
    expect(await screen.findByText('Description for iss-a')).toBeInTheDocument();
    expect(screen.getByLabelText('Current URL')).toHaveTextContent('issueSeverity=high&selectedIssueId=iss-a');
    fireEvent.change(screen.getByRole('textbox', { name: 'Search issues by title' }), {
      target: { value: 'Wrong' },
    });
    expect(await screen.findByText('Description for iss-b')).toBeInTheDocument();
    expect(screen.getByLabelText('Current URL')).toHaveTextContent('selectedIssueId=iss-b');
    expect(screen.getByLabelText('Current URL')).toHaveTextContent('issueSearch=Wrong');
  });
  test('keeps a deep-linked trace open on the issue it was shared with', async () => {
    renderPage('?selectedIssueId=iss-a&selectedEvaluationId=tr-1&selectedIssueSpanId=span-1');

    // The drawer opens over the issue, so the link reopens exactly what was shared.
    expect(await screen.findByRole('dialog')).toBeInTheDocument();
    expect(screen.getByLabelText('Current URL')).toHaveTextContent('selectedEvaluationId=tr-1');
    expect(screen.getByLabelText('Current URL')).toHaveTextContent('selectedIssueSpanId=span-1');
  });
  test('drops a trace left in the URL when the selection moves to another issue', async () => {
    // No selected issue, so the page picks the first one itself. A trace from elsewhere is not
    // part of that issue's evidence, and its drawer must not open over it.
    renderPage('?selectedEvaluationId=tr-1&selectedIssueSpanId=span-1');

    expect(await screen.findByText('Description for iss-a')).toBeInTheDocument();
    expect(screen.getByLabelText('Current URL')).toHaveTextContent('selectedIssueId=iss-a');
    expect(screen.getByLabelText('Current URL')).not.toHaveTextContent('selectedEvaluationId');
    expect(screen.getByLabelText('Current URL')).not.toHaveTextContent('selectedIssueSpanId');
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });
  test.each(['latest', 'older'])('refreshes Issues once when the %s tracked run completes', async (whichRun) => {
    issues = [];
    latestRun = {
      info: { run_id: 'run-1', status: 'RUNNING' },
      data: { tags: [{ key: 'mlflow.issueDetection.jobId', value: 'job-1' }] },
    };
    jobStatus = 'RUNNING';
    let olderJobStatus = 'RUNNING';
    server.use(
      rest.get('*/ajax-api/3.0/mlflow/jobs/job-2', (_req, res, ctx) =>
        res(ctx.json({ status: olderJobStatus, result: { issues: issues.length, total_traces_analyzed: 50 } })),
      ),
    );
    const trackedJobId = whichRun === 'latest' ? 'job-1' : 'job-2';
    recordSubmittedIssueDetectionJob({
      experimentId: '1',
      jobId: trackedJobId,
      runId: whichRun === 'latest' ? 'run-1' : 'run-2',
      traceCount: 50,
      submittedAtMs: Date.now(),
    });
    const { client } = renderPage('', true);
    // While a run is in flight the page shows the progress panel in place of the empty state.
    await screen.findByText('Detecting issues...');
    await waitFor(() => expect(client.isFetching()).toBe(0));
    const invalidate = jest.spyOn(client, 'invalidateQueries');
    issues = [makeIssue('iss-detected', 'Detected by job')];
    if (whichRun === 'latest') {
      jobStatus = 'SUCCEEDED';
      latestRun.info.status = 'FINISHED';
    } else {
      olderJobStatus = 'SUCCEEDED';
    }
    await act(async () => {
      await client.invalidateQueries(['FETCH_JOB_STATUS', trackedJobId]);
    });
    expect(await screen.findByRole('button', { name: /Detected by job/ })).toBeInTheDocument();
    expect(await screen.findByText('Issue detection completed')).toBeInTheDocument();
    await act(async () => {
      await client.invalidateQueries(['LATEST_ISSUE_DETECTION']);
    });
    expect(invalidate.mock.calls.filter(([key]) => Array.isArray(key) && key[0] === 'SEARCH_ISSUES')).toHaveLength(1);
    invalidate.mockRestore();
    // Selecting the detected issue starts loading its evidence traces. Let that land here, so the
    // next test's request handlers only ever see the requests that test made.
    await waitFor(() => expect(client.isFetching()).toBe(0));
  });
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
      expect(screen.queryByRole('button', { name: 'Detect issues in traces' })).not.toBeInTheDocument();
      expect(screen.queryByRole('textbox', { name: 'Search issues by title' })).not.toBeInTheDocument();
      expect(fetchIssueData).not.toHaveBeenCalled();
    },
  );
  test('renders the description without deprecated root causes and preserves selection through back navigation', async () => {
    Object.assign(issues[0], { root_causes: ['Repeated model calls'] });
    renderPage('?selectedIssueId=iss-a');
    expect(await screen.findByText('Description for iss-a')).toBeInTheDocument();
    expect(screen.queryByText('Repeated model calls')).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /Wrong answer/ }));
    expect(await screen.findByText('Description for iss-b')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Back' }));
    expect(await screen.findByText('Description for iss-a')).toBeInTheDocument();
    expect(searches).toContainEqual(
      expect.objectContaining({
        experiment_id: '1',
        view: 'basic',
        include_total_count: true,
        include_trace_count: true,
      }),
    );
  });
  test('loads the next page from SQL and resets pagination when searching', async () => {
    responsePageSize = 2;
    issues = Array.from({ length: 3 }, (_, index) =>
      makeIssue(`iss-${index}`, index === 2 ? 'Later match' : `Issue ${index}`),
    );
    renderPage();
    const loadMore = await screen.findByRole('button', { name: 'Load more' });
    expect(screen.queryByRole('button', { name: /Later match/ })).not.toBeInTheDocument();
    await userEvent.click(loadMore);
    expect(await screen.findByRole('button', { name: /Later match/ })).toBeInTheDocument();
    expect(searches).toContainEqual(expect.objectContaining({ page_token: '2', max_results: 50 }));
    fireEvent.change(screen.getByRole('textbox', { name: 'Search issues by title' }), {
      target: { value: 'Later match' },
    });
    await waitFor(() => expect(screen.queryByRole('button', { name: /Issue 0/ })).not.toBeInTheDocument());
    await waitFor(() =>
      expect(searches[searches.length - 1]).toEqual(expect.objectContaining({ name: 'Later match', max_results: 50 })),
    );
    expect(searches[searches.length - 1]['page_token']).toBeUndefined();
    expect(await screen.findByRole('button', { name: /Later match/ })).toBeInTheDocument();
  });
  test('merges repeated IDs if an already seen Issue moves past the cursor', async () => {
    server.use(
      rest.post('*/ajax-api/3.0/mlflow/issues/search', async (req, res, ctx) => {
        const body = await req.json();
        return res(
          ctx.json(
            body.page_token
              ? { issues: [makeIssue('iss-a', 'Updated title'), makeIssue('iss-b')], total_count: 2 }
              : { issues: [makeIssue('iss-a')], total_count: 2, next_page_token: 'opaque-cursor' },
          ),
        );
      }),
    );
    renderPage();
    await userEvent.click(await screen.findByRole('button', { name: 'Load more' }));
    await screen.findByRole('button', { name: /Updated title/ });
    expect(screen.getAllByTestId('experiment-issue-row')).toHaveLength(2);
  });
  test('searches beyond the loaded page in SQL', async () => {
    responsePageSize = 2;
    issues = Array.from({ length: 3 }, (_, index) =>
      makeIssue(`iss-${index}`, index === 2 ? 'Later match' : `Issue ${index}`),
    );
    renderPage();
    await screen.findByRole('button', { name: /Issue 0/ });
    expect(screen.queryByRole('button', { name: /Later match/ })).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole('textbox', { name: 'Search issues by title' }), {
      target: { value: 'Later match' },
    });
    await waitFor(() =>
      expect(searches).toContainEqual(expect.objectContaining({ name: 'Later match', max_results: 50 })),
    );
    await waitFor(() => expect(screen.queryByRole('button', { name: /Issue 0/ })).not.toBeInTheDocument());
    expect(await screen.findByRole('button', { name: /Later match/ })).toBeInTheDocument();
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
    let rejectPatch!: () => void;
    const patchReady = new Promise<void>((resolve) => {
      rejectPatch = resolve;
    });
    server.use(
      rest.patch('*/ajax-api/3.0/mlflow/issues/:id', async (_req, res, ctx) => {
        await patchReady;
        return res(ctx.status(403), ctx.json({ message: 'Read only' }));
      }),
    );
    renderPage('?selectedIssueId=iss-a');
    const statusButton = await screen.findByRole('button', { name: 'Open' });
    await userEvent.click(statusButton);
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Mark resolved' }));
    await waitFor(() => expect(statusButton).toHaveTextContent('Resolved'));
    rejectPatch();
    expect(await screen.findByText('Unable to update issue')).toBeInTheDocument();
    expect(await screen.findByRole('button', { name: 'Open' })).toBeEnabled();
    expect(issues[0].status).toBe('pending');
  });
  test('resolves and reopens a linked issue even when its row is filtered out', async () => {
    renderPage('?selectedIssueId=iss-a');
    await userEvent.click(await screen.findByRole('button', { name: 'Open' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Mark resolved' }));
    await waitFor(() => expect(issues[0].status).toBe('resolved'));
    await userEvent.click(await screen.findByRole('button', { name: 'Resolved' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Reopen' }));
    await waitFor(() => expect(issues[0].status).toBe('pending'));
    expect(await screen.findByText('Description for iss-a')).toBeInTheDocument();
  });
  test('removes the default Open chip to include resolved issues and changes the server sort', async () => {
    issues[1].status = 'resolved';
    renderPage();
    await screen.findByRole('button', { name: /Slow response/ });
    expect(screen.queryByRole('button', { name: /Wrong answer/ })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Remove Open from Status filter' }));
    expect(await screen.findByRole('button', { name: /Wrong answer/ })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Sort Severity' }));
    await userEvent.click(await screen.findByRole('menuitemradio', { name: 'Most recent' }));
    await waitFor(() =>
      expect(searches).toContainEqual(expect.objectContaining({ statuses: [], order_by: ['created_timestamp DESC'] })),
    );
  });
  test('clears all filter dimensions from the popover while preserving the title search', async () => {
    renderPage('?issueSearch=Slow&issueSeverity=high&issueCategory=latency');
    await screen.findByRole('button', { name: /Slow response/ });
    await userEvent.click(screen.getByRole('button', { name: 'Filter issues' }));
    expect(await screen.findByRole('combobox', { name: 'Category' })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Clear filters' }));
    await waitFor(() =>
      expect(searches).toContainEqual(
        expect.objectContaining({ name: 'Slow', statuses: [], severities: [], categories: [] }),
      ),
    );
    expect(screen.getByRole('textbox', { name: 'Search issues by title' })).toHaveValue('Slow');
  });
  test('lists category options without search and preserves custom-category links', async () => {
    issues[0].categories = ['CustomerSupport'];
    renderPage('?issueCategory=CustomerSupport');
    await userEvent.click(await screen.findByRole('button', { name: 'Filter issues' }));
    await userEvent.click(screen.getByRole('combobox', { name: 'Category' }));
    expect(screen.queryByRole('searchbox')).not.toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'CustomerSupport', selected: true })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('option', { name: 'Correctness' }));
    await waitFor(() =>
      expect(searches).toContainEqual(
        expect.objectContaining({ categories: ['CustomerSupport', 'correctness', '[correctness]'] }),
      ),
    );
    await userEvent.click(screen.getByRole('option', { name: 'CustomerSupport' }));
    await waitFor(() =>
      expect(searches).toContainEqual(expect.objectContaining({ categories: ['correctness', '[correctness]'] })),
    );
  });
  test.each([
    ['newest', 'Most recent', 'created_timestamp DESC'],
    ['updated', 'Last updated', 'last_updated_timestamp DESC'],
  ])('preserves existing sort links (%s)', async (sort, label, order) => {
    renderPage(`?issueSort=${sort}`);
    expect(await screen.findByRole('button', { name: `Sort ${label}` })).toBeInTheDocument();
    await waitFor(() => expect(searches).toContainEqual(expect.objectContaining({ order_by: [order] })));
  });
  test('lets keyboard users select an issue row and reject it from the status menu', async () => {
    renderPage('?selectedIssueId=iss-a');
    const row = await screen.findByRole('button', { name: /Wrong answer/ });
    row.focus();
    await userEvent.keyboard('{Enter}');
    await screen.findByText('Description for iss-b');
    expect(row).toHaveAttribute('aria-pressed', 'true');
    await userEvent.click(screen.getByRole('button', { name: 'Open' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Not an issue' }));
    await waitFor(() => expect(issues[1].status).toBe('rejected'));
    expect(await screen.findByRole('button', { name: 'Not an issue' })).toBeInTheDocument();
  });
  test('opens the existing detector with recent trace IDs without an experiment EDIT grant', async () => {
    renderPage();
    await userEvent.click(await screen.findByRole('button', { name: 'Detect issues in traces' }));
    expect(await screen.findByRole('dialog')).toHaveTextContent('Selected traces: trace-1, trace-2');
  });
  test.each([200, 500])('preserves detection dialog drafts during a trace refetch (%s)', async (status) => {
    const { client } = renderPage();
    await userEvent.click(await screen.findByRole('button', { name: 'Detect issues in traces' }));
    const dialog = await screen.findByRole('dialog');
    await userEvent.type(screen.getByRole('textbox', { name: 'Detection model draft' }), 'chosen-model');
    let finishResponse: () => void;
    const responseReady = new Promise<void>((resolve) => {
      finishResponse = resolve;
    });
    server.use(
      rest.post('*/ajax-api/3.0/mlflow/traces/search', async (_req, res, ctx) => {
        await responseReady;
        return res(ctx.status(status), ctx.json({ traces: [{ trace_id: 'trace-3' }] }));
      }),
    );
    let refetch: Promise<void>;
    act(() => {
      refetch = client.invalidateQueries(['ISSUE_DETECTION_DEFAULT_TRACES']);
    });
    await waitFor(() => expect(client.isFetching(['ISSUE_DETECTION_DEFAULT_TRACES'])).toBe(1));
    expect(screen.getByRole('dialog')).toBe(dialog);
    expect(screen.getByRole('textbox', { name: 'Detection model draft' })).toHaveValue('chosen-model');
    await act(async () => {
      finishResponse();
      await refetch;
    });
    expect(screen.getByRole('dialog')).toBe(dialog);
    expect(screen.getByRole('textbox', { name: 'Detection model draft' })).toHaveValue('chosen-model');
  });
  test.each(['PENDING', 'RUNNING'])(
    'recovers the button spinner after reload and refreshes SQL issues (%s)',
    async (status) => {
      issues = [];
      latestRun = {
        info: { run_id: 'run-1', status: 'RUNNING' },
        data: { tags: [{ key: 'mlflow.issueDetection.jobId', value: 'job-1' }] },
      };
      jobStatus = status;
      const { client } = renderPage();
      await screen.findByText('Detecting issues...');
      await waitFor(() =>
        expect(client.getQueryData(['FETCH_JOB_STATUS', 'job-1'])).toEqual(expect.objectContaining({ status })),
      );
      // The page now owns the run in progress rather than sending the user to the run page.
      expect(screen.getByText('Detecting issues...')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Cancel detection' })).toBeEnabled();
      expect(screen.getByRole('button', { name: 'Go to Traces' })).toBeInTheDocument();
      expect(screen.queryByRole('link', { name: 'View detection run' })).not.toBeInTheDocument();
      // The panel offers no way to start a second run while one is in flight.
      expect(screen.queryByRole('button', { name: 'Run issue detection' })).not.toBeInTheDocument();
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
      issues = [makeIssue('iss-detected', 'Detected by job')];
      jobStatus = 'SUCCEEDED';
      latestRun.info.status = 'FINISHED';
      await act(async () => {
        await client.invalidateQueries(['FETCH_JOB_STATUS', 'job-1']);
      });
      expect(await screen.findByRole('button', { name: /Detected by job/ })).toBeInTheDocument();
      expect(screen.queryByText('Issue detection completed')).not.toBeInTheDocument();
      expect(screen.queryByText('Detecting issues...')).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Detect issues in traces' })).toHaveAttribute('aria-busy', 'false');
      expect(screen.getByRole('button', { name: 'Detect issues in traces' })).toBeEnabled();
    },
  );
  test.each([false, true])(
    'keeps a submitted job active across stale history and remount (empty: %s)',
    async (empty) => {
      if (empty) issues = [];
      latestRun = { info: { run_id: 'old-run', status: 'FINISHED', start_time: 1 }, data: { tags: [] } };
      jobStatus = 'RUNNING';
      const buttonName = empty ? 'Run issue detection' : 'Detect issues in traces';
      const first = renderPage();
      await screen.findByRole('button', { name: buttonName, busy: false });
      act(() => recordSubmittedIssueDetectionJob({ experimentId: '1', runId: 'run-1', jobId: 'job-1', traceCount: 2 }));
      // An empty list hands the whole area to the progress panel; a populated one keeps its
      // toolbar button and shows the run as busy there.
      const expectInProgress = async () => {
        if (empty) {
          expect(await screen.findByText('Detecting issues...')).toBeInTheDocument();
          expect(screen.queryByRole('button', { name: buttonName })).not.toBeInTheDocument();
        } else {
          expect(await screen.findByRole('button', { name: 'Cancel detection' })).toBeEnabled();
          expect(screen.queryByRole('button', { name: buttonName })).not.toBeInTheDocument();
        }
      };
      await expectInProgress();
      await waitFor(() => expect(first.client.isFetching()).toBe(0));
      await expectInProgress();
      first.unmount();
      const { client } = renderPage();
      await expectInProgress();
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
      jobStatus = 'FAILED';
      await act(async () => {
        await client.invalidateQueries(['FETCH_JOB_STATUS', 'job-1']);
      });
      expect(
        await screen.findByRole('button', { name: empty ? 'Start new detection' : buttonName, busy: false }),
      ).toBeEnabled();
    },
  );
  test.each([{ experimentId: 'other' }, { experimentId: '1', workspace: 'other' }])(
    'ignores submissions outside the current scope (%o)',
    async (scope) => {
      renderPage();
      await screen.findByText('Description for iss-a');
      act(() => recordSubmittedIssueDetectionJob({ ...scope, runId: 'run-1', jobId: 'job-1', traceCount: 2 }));
      expect(screen.getByRole('button', { name: 'Detect issues in traces' })).toBeEnabled();
    },
  );
  test.each([undefined, '0', '2'])(
    'uses completed history when a collaborator cannot read the job (count: %s)',
    async (count) => {
      issues = [];
      latestRun = {
        info: { run_id: 'run-1', status: 'FINISHED' },
        data: {
          tags: [
            { key: 'mlflow.issueDetection.jobId', value: 'job-1' },
            ...(count === undefined ? [] : [{ key: 'mlflow.issueDetection.result.issues', value: count }]),
          ],
        },
      };
      server.use(rest.get('*/ajax-api/3.0/mlflow/jobs/job-1', (_req, res, ctx) => res(ctx.status(403))));
      renderPage();
      expect(
        await screen.findByText(count === '0' ? 'No issues detected' : 'No issues to display'),
      ).toBeInTheDocument();
      expect(screen.queryByText('Discover issues automatically')).not.toBeInTheDocument();
      expect(screen.queryByText('Unable to update detection progress')).not.toBeInTheDocument();
    },
  );
  test.each(['populated', 'filtered-empty', 'unfiltered-empty', 'error'])(
    'only checks existence when necessary (%s)',
    async (mode) => {
      if (mode.includes('empty')) issues = [];
      if (mode === 'error') {
        server.use(
          rest.post('*/ajax-api/3.0/mlflow/issues/search', async (req, res, ctx) => {
            searches.push(await req.json());
            return res(ctx.status(500));
          }),
        );
      }
      const { client } = renderPage(mode === 'unfiltered-empty' ? '?issueStatus=all' : '');
      await screen.findByRole('button', {
        name: mode.includes('empty') ? 'Run issue detection' : 'Detect issues in traces',
      });
      await waitFor(() => expect(client.isFetching()).toBe(0));
      expect(searches.filter((body) => body['max_results'] === 1)).toHaveLength(mode === 'filtered-empty' ? 1 : 0);
    },
  );
  test('retries a failed existence check without calling it a completed empty scan', async () => {
    issues = [];
    latestRun = { info: { run_id: 'run-1', status: 'FINISHED' }, data: { tags: [] } };
    server.use(
      rest.post('*/ajax-api/3.0/mlflow/issues/search', async (req, res, ctx) => {
        const body = await req.json();
        return body['max_results'] === 1 ? res(ctx.status(500)) : res(ctx.json({ issues: [] }));
      }),
    );
    renderPage();
    await screen.findByText('Unable to load issues');
    expect(screen.queryByText('No issues to display')).not.toBeInTheDocument();
    server.use(rest.post('*/ajax-api/3.0/mlflow/issues/search', (_req, res, ctx) => res(ctx.json({ issues: [] }))));
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(await screen.findByText('No issues to display')).toBeInTheDocument();
  });
  test('shows category descriptions on chip hover and focus without extra help buttons', async () => {
    renderPage();
    const row = await screen.findByRole('button', { name: /Slow response/ });
    expect(row).toHaveAccessibleDescription(/./);
    expect(within(row).queryByRole('button')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /About category/ })).not.toBeInTheDocument();
    const chip = within(await screen.findByTestId('experiment-issue-detail')).getByTestId('issue-category-tag');
    await userEvent.hover(chip);
    expect(await screen.findByRole('tooltip')).toHaveTextContent(/./);
    await userEvent.keyboard('{Escape}');
    await userEvent.unhover(chip);
    await waitFor(() => expect(screen.queryByRole('tooltip')).not.toBeInTheDocument());
    act(() => chip.focus());
    expect(await screen.findByRole('tooltip')).toHaveTextContent(/./);
    await userEvent.keyboard('{Escape}');
    expect(chip).toHaveFocus();
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
  });
  test('shares the triage label with filters and omits facet counts', async () => {
    renderPage();
    expect(await screen.findByText('2 results')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Filter issues' }));
    await userEvent.click(screen.getByRole('combobox', { name: 'Severity' }));
    expect(screen.getAllByRole('option').map((option) => option.textContent)).toEqual(['All', 'High', 'Medium', 'Low']);
    await userEvent.keyboard('{Escape}');
    await userEvent.click(screen.getByRole('combobox', { name: 'Status' }));
    expect(await screen.findByRole('option', { name: 'All' })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Rejected' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('option', { name: 'Not an issue' }));
    await waitFor(() =>
      expect(searches).toContainEqual(expect.objectContaining({ statuses: ['pending', 'rejected'] })),
    );
    expect(screen.getByText('Status: Not an issue')).toBeInTheDocument();
  });
  test('does not show first-use guidance after launching from the empty page', async () => {
    localStorage.removeItem(`${DEFAULT_DETECT_ISSUES_GUIDANCE_STORAGE_KEY}_v1`);
    issues = [];
    const { client, unmount } = renderPage();
    await userEvent.click(await screen.findByRole('button', { name: 'Run issue detection' }));
    expect(await screen.findByRole('dialog')).toBeInTheDocument();
    issues = [makeIssue('new-issue')];
    await act(async () => {
      await client.invalidateQueries(['SEARCH_ISSUES']);
    });
    expect(await screen.findByRole('button', { name: 'Detect issues in traces' })).toBeInTheDocument();
    expect(screen.queryByText('Detect Issues in Your Traces')).not.toBeInTheDocument();
    unmount();
    renderPage();
    await screen.findByRole('button', { name: 'Detect issues in traces' });
    expect(screen.queryByText('Detect Issues in Your Traces')).not.toBeInTheDocument();
  });
  test('does not offer to cancel the previous run while the next one is being set up', async () => {
    // A finished run keeps its job ID. Opening the modal makes the button busy again while
    // traces load, and gating cancel on the job ID alone would offer to cancel that dead job.
    latestRun = {
      info: { run_id: 'run-done', status: 'FINISHED' },
      data: { tags: [{ key: 'mlflow.issueDetection.jobId', value: 'job-1' }] },
    };
    jobStatus = 'SUCCEEDED';
    server.use(
      rest.post('*/ajax-api/3.0/mlflow/traces/search', (_req, res, ctx) =>
        res(ctx.delay(400), ctx.json({ traces: [{ trace_id: 'tr-1' }] })),
      ),
    );
    renderPage();
    await screen.findByText('Description for iss-a');

    await userEvent.click(screen.getByRole('button', { name: 'Detect issues in traces' }));

    expect(screen.queryByRole('button', { name: 'Cancel detection' })).not.toBeInTheDocument();
  });
  test('spins rather than offering to cancel a run that does not exist yet', async () => {
    // Between opening the modal and the submit returning a job there is nothing to cancel, so
    // the detect button spins instead of offering an action that would have no target.
    renderPage();
    await screen.findByText('Description for iss-a');
    await userEvent.click(screen.getByRole('button', { name: 'Detect issues in traces' }));
    await screen.findByRole('dialog');

    expect(screen.queryByRole('button', { name: 'Cancel detection' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Detect issues in traces' })).toBeInTheDocument();
  });
  test('discovers a newly submitted job without navigating away', async () => {
    renderPage();
    await screen.findByText('Description for iss-a');
    latestRun = {
      info: { run_id: 'run-1', status: 'RUNNING' },
      data: { tags: [{ key: 'mlflow.issueDetection.jobId', value: 'job-1' }] },
    };
    jobStatus = 'RUNNING';
    act(() => recordSubmittedIssueDetectionJob({ experimentId: '1', jobId: 'job-1', runId: 'run-1', traceCount: 2 }));
    // The run exists, so the action becomes cancelling it rather than starting another.
    expect(await screen.findByRole('button', { name: 'Cancel detection' })).toBeEnabled();
    expect(screen.queryByRole('button', { name: 'Detect issues in traces' })).not.toBeInTheDocument();
  });
  test('keeps the button spinning while job status is unavailable, including permission failures', async () => {
    latestRun = {
      info: { run_id: 'run-1', status: 'RUNNING' },
      data: { tags: [{ key: 'mlflow.issueDetection.jobId', value: 'job-1' }] },
    };
    server.use(
      rest.get('*/ajax-api/3.0/mlflow/jobs/job-1', (_req, res, ctx) =>
        res(ctx.delay(100), ctx.status(403), ctx.json({ message: 'Not the job owner' })),
      ),
    );
    renderPage();
    // A status that cannot be read leaves the run in progress -- and is the case the cancel
    // affordance exists for, since the run may have died without reporting it.
    expect(await screen.findByRole('button', { name: 'Cancel detection' })).toBeEnabled();
    await screen.findByText('Unable to update detection progress');
    expect(screen.getByRole('button', { name: 'Cancel detection' })).toBeEnabled();
    expect(screen.queryByRole('button', { name: 'Detect issues in traces' })).not.toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole('button', { name: 'Retry status' })).toBeEnabled());
  });
  test('a failed cancel still refreshes job status, so a finished run stops blocking', async () => {
    // Reported in review. One failed status read stops polling, so this client can be holding
    // a stale RUNNING while the job has already finalized -- and then Cancel is refused with
    // "already finalized" and nothing refreshed, leaving the page blocked until a reload.
    // The refresh now runs in a `finally`, which is what recovers it. The Traces toolbar uses
    // the identical pattern.
    issues = [];
    latestRun = {
      info: { run_id: 'run-1', status: 'RUNNING' },
      data: { tags: [{ key: 'mlflow.issueDetection.jobId', value: 'job-1' }] },
    };
    jobStatus = 'RUNNING';
    renderPage();
    await screen.findByText('Detecting issues...');

    // The job finalized without this client seeing it; cancelling it is therefore refused.
    latestRun.info.status = 'FINISHED';
    jobStatus = 'SUCCEEDED';
    issues = [makeIssue('iss-late', 'Found after the stale cancel')];
    server.use(
      rest.patch('*/ajax-api/3.0/mlflow/jobs/cancel/job-1', (_req, res, ctx) =>
        res(ctx.status(400), ctx.json({ message: 'Job job-1 is already finalized' })),
      ),
    );

    // Both the progress panel and the action bar offer a cancel, so this picks the panel's.
    await userEvent.click(
      document.querySelector('[data-component-id="mlflow.experiment-issues.detection-progress.cancel"]') as HTMLElement,
    );
    await userEvent.click(within(await screen.findByRole('dialog')).getByRole('button', { name: 'Cancel detection' }));

    // The refresh that followed the failed cancel picked up the real state.
    expect(await screen.findByRole('button', { name: /Found after the stale cancel/ })).toBeInTheDocument();
    expect(screen.queryByText('Detecting issues...')).not.toBeInTheDocument();
  });
  test('retries failed job polling and refreshes issues on completion', async () => {
    issues = [];
    latestRun = {
      info: { run_id: 'run-1', status: 'RUNNING' },
      data: { tags: [{ key: 'mlflow.issueDetection.jobId', value: 'job-1' }] },
    };
    jobStatus = 'RUNNING';
    const { client } = renderPage();
    await screen.findByText('Detecting issues...');
    await waitFor(() => expect(client.isFetching(['FETCH_JOB_STATUS', 'job-1'])).toBe(0));
    server.use(rest.get('*/ajax-api/3.0/mlflow/jobs/job-1', (_req, res, ctx) => res(ctx.status(500))));
    await act(async () => {
      await client.invalidateQueries(['FETCH_JOB_STATUS', 'job-1']);
    });
    await screen.findByText('Unable to update detection progress');
    // A failing status poll leaves the run in progress, so the panel stays put.
    expect(screen.getByText('Detecting issues...')).toBeInTheDocument();
    issues = [makeIssue('iss-recovered', 'Found after retry')];
    latestRun.info.status = 'FINISHED';
    server.use(
      rest.get('*/ajax-api/3.0/mlflow/jobs/job-1', (_req, res, ctx) => res(ctx.json({ status: 'SUCCEEDED' }))),
    );
    await userEvent.click(screen.getByRole('button', { name: 'Retry status' }));
    expect(await screen.findByRole('button', { name: /Found after retry/ })).toBeInTheDocument();
    expect(screen.queryByText('Issue detection completed')).not.toBeInTheDocument();
    expect(screen.queryByText('Detecting issues')).not.toBeInTheDocument();
    expect(screen.queryByText('Unable to update detection progress')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Detect issues in traces' })).toHaveAttribute('aria-busy', 'false');
    expect(screen.getByRole('button', { name: 'Detect issues in traces' })).toBeEnabled();
  });
  test.each([false, true])(
    'refreshes issues once when the run completes with unavailable job status (cached running status: %s)',
    async (cachedJob) => {
      issues = [];
      latestRun = {
        info: { run_id: 'run-1', status: 'RUNNING' },
        data: { tags: [{ key: 'mlflow.issueDetection.jobId', value: 'job-1' }] },
      };
      jobStatus = 'RUNNING';
      if (!cachedJob) {
        server.use(rest.get('*/ajax-api/3.0/mlflow/jobs/job-1', (_req, res, ctx) => res(ctx.status(403))));
      }
      const { client } = renderPage();
      await screen.findByText('Detecting issues...');
      if (cachedJob) {
        await waitFor(() => expect(client.isFetching(['FETCH_JOB_STATUS', 'job-1'])).toBe(0));
        server.use(rest.get('*/ajax-api/3.0/mlflow/jobs/job-1', (_req, res, ctx) => res(ctx.status(500))));
        await act(async () => {
          await client.invalidateQueries(['FETCH_JOB_STATUS', 'job-1']);
        });
      }
      await screen.findByText('Unable to update detection progress');
      issues = [makeIssue('iss-other-user', 'Found by another user')];
      latestRun.info.status = 'FINISHED';
      await act(async () => {
        await client.invalidateQueries(['LATEST_ISSUE_DETECTION']);
      });
      expect(await screen.findByRole('button', { name: /Found by another user/ })).toBeInTheDocument();
      expect(screen.queryByText('Issue detection completed')).not.toBeInTheDocument();
      expect(screen.queryByText('Detecting issues')).not.toBeInTheDocument();
      expect(screen.queryByText('Unable to update detection progress')).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Detect issues in traces' })).toHaveAttribute('aria-busy', 'false');
      expect(screen.getByRole('button', { name: 'Detect issues in traces' })).toBeEnabled();
      const searchCount = searches.length;
      await act(async () => {
        await client.invalidateQueries(['LATEST_ISSUE_DETECTION']);
      });
      expect(searches).toHaveLength(searchCount);
    },
  );
  test('distinguishes a missing Issue from denied access', async () => {
    const { unmount } = renderPage('?selectedIssueId=iss-missing');
    expect(await screen.findByText('Issue not found in this experiment')).toBeInTheDocument();
    unmount();
    server.use(rest.get('*/ajax-api/3.0/mlflow/issues/:id', (_req, res, ctx) => res(ctx.status(403))));
    renderPage('?selectedIssueId=iss-a');
    expect(await screen.findByText('You do not have permission to view these issues')).toBeInTheDocument();
  });
  test.each(['?issueSearch=no-match', ''])('retains filters when existing issues are hidden (%s)', async (query) => {
    issues.forEach((issue) => {
      issue.status = 'resolved';
    });
    renderPage(query);
    expect(await screen.findByText('No issues match these filters.')).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: 'Search issues by title' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Filter issues' })).toBeInTheDocument();
    expect(screen.queryByText('Discover issues automatically')).not.toBeInTheDocument();
  });
  test('shows the empty state without list controls and opens detection from its primary action', async () => {
    issues = [];
    renderPage();
    expect(await screen.findByText('Discover issues automatically')).toBeInTheDocument();
    expect(screen.queryByRole('toolbar')).not.toBeInTheDocument();
    expect(screen.queryByRole('textbox', { name: 'Search issues by title' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Filter issues' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Sort / })).not.toBeInTheDocument();
    expect(screen.queryByText('Status:')).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Learn how it works' })).toHaveAttribute(
      'href',
      'https://mlflow.org/docs/latest/genai/eval-monitor/ai-insights/detect-issues/',
    );
    await userEvent.click(screen.getByRole('button', { name: 'Run issue detection' }));
    expect(await screen.findByRole('dialog')).toHaveTextContent('Selected traces: trace-1, trace-2');
  });
  test.each(['FAILED', 'TIMEOUT'])('offers a centered new detection after an empty %s run', async (status) => {
    issues = [];
    latestRun = {
      info: { run_id: 'run-1', status: 'RUNNING' },
      data: { tags: [{ key: 'mlflow.issueDetection.jobId', value: 'job-1' }] },
    };
    jobStatus = status;
    renderPage();
    const restart = await screen.findByRole('button', { name: 'Start new detection' });
    expect(screen.queryByRole('toolbar')).not.toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.queryByText('No issues detected')).not.toBeInTheDocument();
    expect(
      screen.getByText(
        status === 'TIMEOUT'
          ? 'The previous issue detection timed out. Start a new detection to try again.'
          : 'The previous issue detection failed. Start a new detection to try again.',
      ),
    ).toBeInTheDocument();
    await userEvent.click(restart);
    expect(await screen.findByRole('dialog')).toBeInTheDocument();
  });
  test('retries unavailable detection history from the empty page', async () => {
    issues = [];
    server.use(rest.post('*/ajax-api/2.0/mlflow/runs/search', (_req, res, ctx) => res(ctx.status(500))));
    renderPage();
    expect(await screen.findByText('Unable to load detection history')).toBeInTheDocument();
    server.use(rest.post('*/ajax-api/2.0/mlflow/runs/search', (_req, res, ctx) => res(ctx.json({ runs: [] }))));
    await userEvent.click(screen.getByRole('button', { name: 'Retry status' }));
    await waitFor(() => expect(screen.queryByText('Unable to load detection history')).not.toBeInTheDocument());
    expect(screen.getByRole('button', { name: 'Run issue detection' })).toBeEnabled();
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
        await screen.findByText(
          'Issue detection did not find problems in the analyzed traces. Run it again as your application changes.',
        ),
      ).toBeInTheDocument();
    } else {
      expect(
        await screen.findByText('The previous issue detection failed. Start a new detection to try again.'),
      ).toBeInTheDocument();
      expect(
        screen.queryByText(
          'Issue detection did not find problems in the analyzed traces. Run it again as your application changes.',
        ),
      ).not.toBeInTheDocument();
    }
  });
  test.each(['SUCCEEDED', 'FAILED', 'TIMEOUT', 'CANCELED'])(
    'stops spinning and allows another detection after %s',
    async (status) => {
      latestRun = {
        info: { run_id: 'run-1', status: 'RUNNING' },
        data: { tags: [{ key: 'mlflow.issueDetection.jobId', value: 'job-1' }] },
      };
      jobStatus = 'RUNNING';
      const { client } = renderPage();
      expect(await screen.findByRole('button', { name: 'Cancel detection' })).toBeEnabled();
      await waitFor(() => expect(client.isFetching(['FETCH_JOB_STATUS', 'job-1'])).toBe(0));
      jobStatus = status;
      await act(async () => {
        await client.invalidateQueries(['FETCH_JOB_STATUS', 'job-1']);
      });
      // A terminal run hands the action back to starting the next one.
      expect(screen.queryByRole('button', { name: 'Cancel detection' })).not.toBeInTheDocument();
      const detectButton = await screen.findByRole('button', { name: 'Detect issues in traces' });
      expect(detectButton).toHaveAttribute('aria-busy', 'false');
      expect(detectButton).toBeEnabled();
      expect(screen.queryByRole('link', { name: 'View detection run' })).not.toBeInTheDocument();
      await userEvent.click(detectButton);
      expect(await screen.findByRole('dialog')).toBeInTheDocument();
    },
  );
  test('does not call Issue APIs for unsupported or proxied storage', async () => {
    server.use(rest.get('*/ajax-api/3.0/mlflow/server-info', (_req, res, ctx) => res(ctx.json({ store_type: null }))));
    renderPage();
    expect(await screen.findByText('The Issues tab requires an MLflow server with SQL storage.')).toBeInTheDocument();
    expect(searches).toEqual([]);
  });
});
