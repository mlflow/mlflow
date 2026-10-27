import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { rest } from 'msw';
import userEvent, { PointerEventsCheckLevel } from '@testing-library/user-event';

import { shouldEnableExperimentIssuesTab } from '../../../common/utils/FeatureUtils';
import { MemoryRouter, Route, Routes } from '../../../common/utils/RoutingUtils';
import { QueryClient, QueryClientProvider } from '../../../common/utils/reactQueryHooks';
import { setupServer } from '../../../common/utils/setup-msw';
import { renderWithDesignSystem, screen, waitFor, within } from '../../../common/utils/TestUtils.react18';
import {
  MLFLOW_ISSUE_DETECTION_ERROR_TAG,
  MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE_TAG,
  MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG,
  MLFLOW_ISSUE_DETECTION_WINDOW_END_TAG,
  MLFLOW_ISSUE_DETECTION_WINDOW_START_TAG,
} from '../../constants';
import ExperimentIssueScheduledRunsPage from './ExperimentIssueScheduledRunsPage';
import { ISSUE_DETECTION_SCHEDULE_VERSION } from './issueDetectionSchedule';

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableExperimentIssuesTab: jest.fn(),
}));

const DAY_MS = 24 * 60 * 60 * 1000;

/**
 * The clock is left alone, and each kind of assertion gets a timestamp it can be deterministic
 * about.
 *
 * Freezing it was the obvious move -- `TimeAgo` measures with `new Date()`, so a fixture dated in
 * the future reads as "just now" -- but fake timers kept React flushing work after the test ended,
 * which crashed the suite in a torn-down jsdom. So anything compared against now is built from the
 * real now instead, while the schedule's instants, which are formatted from the values they are
 * given rather than from the clock, stay fixed and can be written out.
 */
let nowMs: number;
/** A fixed covered window, formatted from these bounds whatever today is. */
const WINDOW_START_MS = Date.UTC(2027, 0, 1, 12, 0);
const WINDOW_END_MS = Date.UTC(2027, 0, 8, 12, 0);
/** The schedule's own next fire time: a Monday at 21:15 UTC, as the schedule tag says. */
const NEXT_FIRE_MS = Date.UTC(2027, 0, 18, 21, 15);

interface StoredRun {
  runUuid: string;
  status: string;
  startTime: number;
  endTime?: number;
  tags: Record<string, string>;
}

let experimentTags: { key: string; value: string }[];
let storedRuns: StoredRun[];
/** `source_run_id` of each issue, which is what the new-issue count is derived from. */
let issueSourceRunIds: string[];
let issueSearchFails: boolean;
let runSearchPages: number;
/** Filters the page asked runs with, so the search can be asserted rather than assumed. */
let runFilters: string[];
/** Set to fail the run search, which is how a read failure is exercised. */
let runSearchFails: boolean;

const server = setupServer(
  rest.get('*/ajax-api/3.0/mlflow/server-info', (_req, res, ctx) =>
    res(ctx.json({ store_type: 'SqlStore', features_enabled: { gateway: true, scheduled_issue_detection: true } })),
  ),
  rest.get('*/ajax-api/2.0/mlflow/experiments/get', (_req, res, ctx) =>
    res(ctx.json({ experiment: { experiment_id: '1', name: 'exp', tags: experimentTags } })),
  ),
  rest.post('*/ajax-api/2.0/mlflow/runs/search', async (req, res, ctx) => {
    if (runSearchFails) {
      return res(ctx.status(500), ctx.json({ message: 'nope' }));
    }
    const body = (await req.json()) as { filter?: string; page_token?: string };
    runFilters.push(body.filter ?? '');
    const page = body.page_token ? Number(body.page_token) : 0;
    return res(
      ctx.json({
        runs: storedRuns.map(({ runUuid, status, startTime, endTime, tags }) => ({
          info: {
            run_uuid: runUuid,
            run_id: runUuid,
            status,
            start_time: startTime,
            ...(endTime === undefined ? {} : { end_time: endTime }),
          },
          data: { tags: Object.entries(tags).map(([key, value]) => ({ key, value })) },
        })),
        ...(page + 1 < runSearchPages ? { next_page_token: String(page + 1) } : {}),
      }),
    );
  }),
  rest.get('*/ajax-api/3.0/mlflow/gateway/endpoints/list', (_req, res, ctx) => res(ctx.json({ endpoints: [] }))),
  rest.get('*/ajax-api/3.0/mlflow/gateway/secrets/list', (_req, res, ctx) =>
    res(ctx.json({ secrets: [{ secret_id: 'sec-1', secret_name: 'key', provider: 'openai' }] })),
  ),
  rest.get('*/ajax-api/3.0/mlflow/gateway/provider-config', (_req, res, ctx) =>
    res(ctx.json({ auth_modes: [], default_mode: 'api_key', env_credentials_available: true })),
  ),
  rest.post('*/ajax-api/3.0/mlflow/issues/search', async (req, res, ctx) => {
    if (issueSearchFails) {
      return res(ctx.status(500), ctx.json({ message: 'nope' }));
    }
    const body = (await req.json()) as { filter_string?: string };
    const runId = /source_run_id = '([^']*)'/.exec(body.filter_string ?? '')?.[1];
    return res(ctx.json({ issues: [], total_count: issueSourceRunIds.filter((source) => source === runId).length }));
  }),
);

const scheduleTag = (cadence: string | null, { paused = false } = {}) => ({
  key: MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG,
  value: JSON.stringify({
    version: ISSUE_DETECTION_SCHEDULE_VERSION,
    cadence,
    timezone: 'UTC',
    paused,
    fireTime: '21:15',
    weekday: 0,
  }),
});

const scheduleStateTag = (nextFireTimeMs: number) => ({
  key: MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE_TAG,
  value: JSON.stringify({
    version: ISSUE_DETECTION_SCHEDULE_VERSION,
    lastFireTimeMs: 1,
    timing: 'weekly|21:15|0|UTC',
    nextFireTimeMs,
  }),
});

const succeeded = (overrides: Partial<StoredRun> = {}): StoredRun => ({
  runUuid: 'run-ok',
  status: 'FINISHED',
  startTime: nowMs - 7 * DAY_MS,
  endTime: nowMs - 7 * DAY_MS + 192_000,
  tags: {
    total_traces: '312',
    total_cost_usd: '0.42',
    model: 'openai:/gpt-5-mini',
    [MLFLOW_ISSUE_DETECTION_WINDOW_START_TAG]: String(WINDOW_START_MS),
    [MLFLOW_ISSUE_DETECTION_WINDOW_END_TAG]: String(WINDOW_END_MS),
  },
  ...overrides,
});

/**
 * Held so it can be torn down between tests.
 *
 * The page issues one request per occurrence for its issue counts. Left in flight past the end of
 * a test they settle against a torn-down renderer, which jest reports as a log after the test
 * finished -- intermittently, and attributed to whichever suite happened to be running.
 */
let queryClient: QueryClient;

const renderPage = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, cacheTime: 0 } } });
  queryClient = client;
  return renderWithDesignSystem(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/experiments/1/issues/scheduled-runs']}>
        <Routes>
          <Route
            path="/experiments/:experimentId/issues/scheduled-runs"
            element={<ExperimentIssueScheduledRunsPage />}
          />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
};

/** The row an occurrence rendered, found by the model cell so the outcome can be asserted on it. */
const rowFor = async (text: string | RegExp) => (await screen.findByText(text)).closest('[role="row"]') as HTMLElement;

beforeEach(() => {
  jest.mocked(shouldEnableExperimentIssuesTab).mockReturnValue(true);
  nowMs = Date.now();
  experimentTags = [scheduleTag('weekly'), scheduleStateTag(NEXT_FIRE_MS)];
  storedRuns = [succeeded()];
  issueSourceRunIds = [];
  issueSearchFails = false;
  runSearchPages = 1;
  runFilters = [];
  runSearchFails = false;
});

afterEach(() => {
  queryClient?.cancelQueries();
  queryClient?.clear();
});

describe('ExperimentIssueScheduledRunsPage', () => {
  test('asks only for the occurrences a schedule fired', async () => {
    renderPage();

    await waitFor(() => expect(runFilters.length).toBeGreaterThan(0));
    // Searching by run type instead would list every manual detection run as a scheduled one.
    expect(runFilters[0]).toBe("tags.`mlflow.issueDetection.trigger` = 'schedule'");
  });

  test('describes the schedule and when it next fires', async () => {
    renderPage();

    expect(await screen.findByText('Weekly on Monday at 21:15 (UTC)')).toBeInTheDocument();
    expect(await screen.findByText(/Jan 18, 2027, 21:15/)).toBeInTheDocument();
  });

  test('a paused schedule says so once, and names no next run', async () => {
    experimentTags = [scheduleTag('weekly', { paused: true })];
    renderPage();

    // Said under the title, where the cadence would otherwise be. The Next run figure shows a
    // dash rather than repeating it, which is what having both read like two separate facts.
    expect(await screen.findByText('Schedule is off')).toBeInTheDocument();
    expect(screen.getByText('Next run').parentElement).toHaveTextContent('—');
  });

  test('shows what the occurrence did', async () => {
    issueSourceRunIds = ['run-ok', 'run-ok'];
    renderPage();

    const row = await rowFor('openai:/gpt-5-mini');
    expect(await within(row).findByText('Succeeded')).toBeInTheDocument();
    expect(within(row).getByText('312')).toBeInTheDocument();
    expect(within(row).getByText('$0.42')).toBeInTheDocument();
    expect(within(row).getByText('3m 12s')).toBeInTheDocument();
    expect(within(row).getByText('7 days ago')).toBeInTheDocument();
    // Both bounds: the end is the fire time rather than when the run began, and the start is
    // wherever the last occurrence left its watermark -- which after a quiet or failed occurrence
    // reaches further back than one cadence interval.
    // No year on either bound: a window reaches back eight days at most, so printing 2027 twice
    // per row only made the column wide enough to push the rest of the table off screen.
    expect(row).toHaveTextContent(/Jan 1, 12:00\s*→\s*Jan 8, 12:00/);
    await waitFor(() => expect(within(row).getByText('2')).toBeInTheDocument());
  });

  test('an occurrence that analyzed nothing is not reported as a success', async () => {
    // A finished run over zero traces fired over a window with nothing in it. "Succeeded" would
    // claim an analysis that never happened.
    storedRuns = [
      succeeded({
        runUuid: 'run-quiet',
        tags: { total_traces: '0', total_cost_usd: '0.0', model: 'openai:/gpt-5-mini' },
      }),
    ];
    renderPage();

    const row = await rowFor('openai:/gpt-5-mini');
    expect(await within(row).findByText('Nothing to analyze')).toBeInTheDocument();
    expect(within(row).queryByText('Succeeded')).not.toBeInTheDocument();
  });

  test('a cancelled occurrence is told apart from one that broke', async () => {
    storedRuns = [succeeded({ runUuid: 'run-killed', status: 'KILLED' })];
    renderPage();

    const row = await rowFor('openai:/gpt-5-mini');
    expect(await within(row).findByText('Cancelled')).toBeInTheDocument();
  });

  test('a failed occurrence reports as failed and claims no cost', async () => {
    storedRuns = [
      succeeded({
        runUuid: 'run-bad',
        status: 'FAILED',
        tags: {
          total_traces: '184',
          model: 'openai:/gpt-5-mini',
          [MLFLOW_ISSUE_DETECTION_ERROR_TAG]: 'MlflowException: no API key available',
        },
      }),
    ];
    renderPage();

    const row = await rowFor('openai:/gpt-5-mini');
    expect(await within(row).findByText('Failed')).toBeInTheDocument();
    expect(within(row).queryByText(/^\$/)).not.toBeInTheDocument();
    // The provider's own wording, which is the part worth acting on. Without it the row says only
    // that something broke, and the exception reaches nowhere an experiment's owner can read it.
    expect(await screen.findByText('Why it failed')).toBeInTheDocument();
    expect(await screen.findByText('MlflowException: no API key available')).toBeInTheDocument();
  });

  test('an occurrence that failed before the reason was recorded shows no empty detail row', async () => {
    storedRuns = [
      succeeded({ runUuid: 'run-bad', status: 'FAILED', tags: { total_traces: '1', model: 'openai:/gpt-5-mini' } }),
    ];
    renderPage();

    const row = await rowFor('openai:/gpt-5-mini');
    expect(await within(row).findByText('Failed')).toBeInTheDocument();
    expect(screen.queryByText('Why it failed')).not.toBeInTheDocument();
  });

  test('an occurrence still going claims no issue count yet', async () => {
    // Its count so far would read as a result; it has not finished filing.
    issueSourceRunIds = [];
    storedRuns = [succeeded({ runUuid: 'run-live', status: 'RUNNING', endTime: undefined })];
    renderPage();

    const row = await rowFor('openai:/gpt-5-mini');
    await waitFor(() => expect(within(row).getByText('Running')).toBeInTheDocument());
    expect(within(row).queryByText('0')).not.toBeInTheDocument();
  });

  test('an occurrence still going shows no cost and no issue count', async () => {
    storedRuns = [
      succeeded({ runUuid: 'run-live', status: 'RUNNING', startTime: nowMs - 124_000, endTime: undefined }),
    ];
    renderPage();

    const row = await rowFor('openai:/gpt-5-mini');
    expect(await within(row).findByText('Running')).toBeInTheDocument();
    // Measured against now, so a run with no end time does not read as having started in 1970.
    expect(within(row).getByText(/^2m 0[45]s$/)).toBeInTheDocument();
  });

  test('totals only the occurrences inside the summary window', async () => {
    storedRuns = [succeeded(), succeeded({ runUuid: 'run-old', startTime: nowMs - 40 * DAY_MS })];
    renderPage();

    await waitFor(() => expect(screen.getByText('Runs, last 30 days').parentElement).toHaveTextContent('1'));
    // Every scoped figure names its window: all three are over the same 30 days, and a bare
    // "Traces scanned" read as an all-time count beside two that said otherwise.
    expect(screen.getByText('Traces scanned, last 30 days').parentElement).toHaveTextContent('312');
    expect(screen.getByText('Spend, last 30 days').parentElement).toHaveTextContent('$0.42');
  });

  test('an issue count that cannot be read leaves the cell blank rather than reading zero', async () => {
    // Zero would say the occurrence found nothing, which is a different claim from "unknown".
    issueSearchFails = true;
    renderPage();

    const row = await rowFor('openai:/gpt-5-mini');
    await waitFor(() => expect(within(row).getByText('Succeeded')).toBeInTheDocument());
    expect(within(row).queryByText('0')).not.toBeInTheDocument();
  });

  test('hedges the count while more pages remain', async () => {
    runSearchPages = 3;
    renderPage();

    // A run search returns no total, so an exact count would be a guess stated as a fact.
    expect(await screen.findByText(/\+ runs loaded/)).toBeInTheDocument();
  });

  test('states the count exactly once every page is loaded', async () => {
    renderPage();

    expect(await screen.findByText('1 run')).toBeInTheDocument();
  });

  test('a schedule with no occurrences yet names when the first one runs', async () => {
    storedRuns = [];
    renderPage();

    expect(await screen.findByText('No occurrences yet')).toBeInTheDocument();
    expect(await screen.findByText(/The first one runs Jan 18, 2027, 21:15/)).toBeInTheDocument();
  });

  test('an empty page shows the state on its own, with no figures or column headers', async () => {
    // The figures are all dashes and zeroes with nothing to total, and a column header over no
    // rows labels nothing, so neither earns its place until there is a row to describe.
    experimentTags = [scheduleTag('weekly', { paused: true })];
    storedRuns = [];
    renderPage();

    expect(await screen.findByText('This schedule is off')).toBeInTheDocument();
    expect(await screen.findByText('Turn the schedule on and the runs it fires will appear here.')).toBeInTheDocument();
    expect(screen.queryByText('Runs, last 30 days')).not.toBeInTheDocument();
    expect(screen.queryByText('Spend, last 30 days')).not.toBeInTheDocument();
    expect(screen.queryByText('Window covered')).not.toBeInTheDocument();
    expect(screen.queryByText('Scanned traces')).not.toBeInTheDocument();
    // And the schedule's own line, which would only repeat the title above.
    expect(screen.queryByText('Schedule is off')).not.toBeInTheDocument();
  });

  test('an empty page for a live schedule keeps its cadence line', async () => {
    // Nothing duplicated there: the title says no occurrences yet, the line says what it will do.
    storedRuns = [];
    renderPage();

    expect(await screen.findByText('No occurrences yet')).toBeInTheDocument();
    expect(screen.getByText('Weekly on Monday at 21:15 (UTC)')).toBeInTheDocument();
    expect(screen.queryByText('Runs, last 30 days')).not.toBeInTheDocument();
  });

  test('opens the schedule dialog without leaving the page', async () => {
    // Editing used to mean navigating back to the Issues tab to find the dialog there.
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderPage();

    // Disabled until the schedule has loaded, so the dialog never opens against unknown state.
    await waitFor(async () => expect(await screen.findByRole('button', { name: 'Edit schedule' })).not.toBeDisabled());
    await user.click(await screen.findByRole('button', { name: 'Edit schedule' }));

    expect(await screen.findByRole('button', { name: 'Save' })).toBeInTheDocument();
    expect(screen.getByText('Scheduled detection runs')).toBeInTheDocument();
  });

  test('a failed read says so instead of claiming the schedule never fired', async () => {
    // Left to the empty state, a transient outage would tell someone their schedule has no
    // history -- the one thing this page exists to answer.
    runSearchFails = true;
    renderPage();

    expect(await screen.findByText("Could not load this schedule's history")).toBeInTheDocument();
    expect(screen.queryByText('No occurrences yet')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
  });

  test('a paused schedule names no next run even when its dispatch state still has one', async () => {
    // The state tag keeps the fire time it was last given, and the dispatcher will not honour it
    // while paused, so naming it beside "Schedule is off" would promise a run that never comes.
    experimentTags = [scheduleTag('weekly', { paused: true }), scheduleStateTag(NEXT_FIRE_MS)];
    renderPage();

    expect(await screen.findByText('Schedule is off')).toBeInTheDocument();
    expect(screen.getByText('Next run').parentElement).toHaveTextContent('—');
    expect(screen.queryByText(/Jan 18, 2027/)).not.toBeInTheDocument();
  });

  test('the 30-day figures are marked as lower bounds while pages remain', async () => {
    // They total the rows loaded so far, so presenting them as exact totals would understate a
    // schedule with more occurrences than one page.
    runSearchPages = 3;
    renderPage();

    await waitFor(() => expect(screen.getByText('Runs, last 30 days').parentElement).toHaveTextContent('1+'));
    expect(screen.getByText('Traces scanned, last 30 days').parentElement).toHaveTextContent('312+');
    expect(screen.getByText('Spend, last 30 days').parentElement).toHaveTextContent('$0.42+');
  });

  test('the figures are exact once every page is loaded', async () => {
    renderPage();

    await waitFor(() => expect(screen.getByText('Runs, last 30 days').parentElement).toHaveTextContent('1'));
    expect(screen.getByText('Runs, last 30 days').parentElement).not.toHaveTextContent('1+');
  });

  test('is not reachable while the Issues tab is off', async () => {
    jest.mocked(shouldEnableExperimentIssuesTab).mockReturnValue(false);
    renderPage();

    expect(await screen.findByText(/Page not found/i)).toBeInTheDocument();
  });
});
