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
  MLFLOW_ISSUE_DETECTION_TRIGGER_TAG,
  MLFLOW_ISSUE_DETECTION_WINDOW_END_TAG,
  MLFLOW_ISSUE_DETECTION_WINDOW_START_TAG,
} from '../../constants';
import ExperimentIssueDetectionRunsPage from './ExperimentIssueDetectionRunsPage';
import { ISSUE_DETECTION_SCHEDULE_VERSION } from './issueDetectionSchedule';
import { DETECTION_RUNS_PAGE_SIZE } from './useDetectionRuns';

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
/** Filters the page asked runs with, so the search can be asserted rather than assumed. */
let runFilters: string[];
/** Set to fail the run search, which is how a read failure is exercised. */
let runSearchFails: boolean;
/** How many run searches were refused, so a test can wait for the failure it caused. */
let runSearchFailures: number;

const server = setupServer(
  rest.get('*/ajax-api/3.0/mlflow/server-info', (_req, res, ctx) =>
    res(ctx.json({ store_type: 'SqlStore', features_enabled: { gateway: true, scheduled_issue_detection: true } })),
  ),
  rest.get('*/ajax-api/2.0/mlflow/experiments/get', (_req, res, ctx) =>
    res(ctx.json({ experiment: { experiment_id: '1', name: 'exp', tags: experimentTags } })),
  ),
  rest.post('*/ajax-api/2.0/mlflow/runs/search', async (req, res, ctx) => {
    if (runSearchFails) {
      runSearchFailures += 1;
      return res(ctx.status(500), ctx.json({ message: 'nope' }));
    }
    const body = (await req.json()) as { filter?: string; page_token?: string; max_results?: number };
    runFilters.push(body.filter ?? '');
    // The store's own paging, so the token the page sends back decides what it is answered with.
    const offset = body.page_token ? Number(body.page_token) : 0;
    const pageSize = body.max_results ?? storedRuns.length;
    const nextOffset = offset + pageSize;
    return res(
      ctx.json({
        runs: storedRuns.slice(offset, nextOffset).map(({ runUuid, status, startTime, endTime, tags }) => ({
          info: {
            run_uuid: runUuid,
            run_id: runUuid,
            status,
            start_time: startTime,
            ...(endTime === undefined ? {} : { end_time: endTime }),
          },
          data: { tags: Object.entries(tags).map(([key, value]) => ({ key, value })) },
        })),
        ...(nextOffset < storedRuns.length ? { next_page_token: String(nextOffset) } : {}),
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

/** A scheduled occurrence that finished: the tags the scheduled job writes, window included. */
const scheduled = (overrides: Partial<StoredRun> = {}): StoredRun => ({
  runUuid: 'run-ok',
  status: 'FINISHED',
  startTime: nowMs - 7 * DAY_MS,
  endTime: nowMs - 7 * DAY_MS + 192_000,
  tags: {
    [MLFLOW_ISSUE_DETECTION_TRIGGER_TAG]: 'schedule',
    // What was handed in, and what the sampler left to analyze. The column shows the second.
    total_traces: '640',
    total_traces_analyzed: '312',
    total_cost_usd: '0.42',
    model: 'openai:/gpt-5-mini',
    [MLFLOW_ISSUE_DETECTION_WINDOW_START_TAG]: String(WINDOW_START_MS),
    [MLFLOW_ISSUE_DETECTION_WINDOW_END_TAG]: String(WINDOW_END_MS),
  },
  ...overrides,
});

/** A run somebody started from the Issues tab: no trigger tag, and no window to record. */
const adhoc = (overrides: Partial<StoredRun> = {}): StoredRun => ({
  runUuid: 'run-adhoc',
  status: 'FINISHED',
  startTime: nowMs - 4 * 60 * 60 * 1000,
  endTime: nowMs - 4 * 60 * 60 * 1000 + 71_000,
  tags: { total_traces_analyzed: '40', total_cost_usd: '0.08', model: 'openai:/gpt-5.6-sol' },
  ...overrides,
});

/** Enough runs to fill a page and start another, each nameable by its model. */
const manyRuns = (count: number): StoredRun[] =>
  Array.from({ length: count }, (_, index) =>
    adhoc({ runUuid: `run-${index}`, tags: { total_traces_analyzed: '1', model: `openai:/m-${index}` } }),
  );

/**
 * Held so it can be torn down between tests.
 *
 * The page issues one request per run for its issue counts. Left in flight past the end of a test
 * they settle against a torn-down renderer, which jest reports as a log after the test finished --
 * intermittently, and attributed to whichever suite happened to be running.
 */
let queryClient: QueryClient;

const renderPage = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, cacheTime: 0 } } });
  queryClient = client;
  return renderWithDesignSystem(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/experiments/1/issues/runs']}>
        <Routes>
          <Route path="/experiments/:experimentId/issues/runs" element={<ExperimentIssueDetectionRunsPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
};

/** The row a run rendered, found by the model cell so the rest of the row can be asserted on it. */
const rowFor = async (text: string | RegExp) => (await screen.findByText(text)).closest('[role="row"]') as HTMLElement;

beforeEach(() => {
  jest.mocked(shouldEnableExperimentIssuesTab).mockReturnValue(true);
  nowMs = Date.now();
  experimentTags = [scheduleTag('weekly'), scheduleStateTag(NEXT_FIRE_MS)];
  storedRuns = [scheduled()];
  issueSourceRunIds = [];
  issueSearchFails = false;
  runFilters = [];
  runSearchFails = false;
  runSearchFailures = 0;
});

afterEach(() => {
  queryClient?.cancelQueries();
  queryClient?.clear();
});

describe('ExperimentIssueDetectionRunsPage', () => {
  test('asks for every detection run, however it was started', async () => {
    renderPage();

    await waitFor(() => expect(runFilters.length).toBeGreaterThan(0));
    // Searching on the trigger tag instead would find the scheduled runs and miss the ad-hoc
    // ones, which have no such tag at all.
    expect(runFilters[0]).toBe("tags.`mlflow.runType` = 'issue_detection'");
  });

  test('describes the schedule and when it next fires', async () => {
    renderPage();

    expect(await screen.findByText(/Scheduled weekly on Monday at 21:15 \(UTC\)/)).toBeInTheDocument();
    expect(await screen.findByText(/Jan 18, 2027, 21:15/)).toBeInTheDocument();
  });

  test('a paused schedule says so, and names no next run', async () => {
    experimentTags = [scheduleTag('weekly', { paused: true })];
    renderPage();

    expect(await screen.findByText('Schedule is off')).toBeInTheDocument();
    expect(screen.queryByText(/Next run/)).not.toBeInTheDocument();
  });

  test('shows what a scheduled run did', async () => {
    issueSourceRunIds = ['run-ok', 'run-ok'];
    renderPage();

    const row = await rowFor('openai:/gpt-5-mini');
    expect(await within(row).findByText('Finished')).toBeInTheDocument();
    expect(within(row).getByText('Scheduled')).toBeInTheDocument();
    // The count the run recorded when it finished, not the 640 it was handed: the sampler caps
    // what it analyzes, so reporting the submitted count would overstate what was looked at.
    expect(within(row).getByText('312')).toBeInTheDocument();
    expect(within(row).queryByText('640')).not.toBeInTheDocument();
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

  test('an ad-hoc run is told apart from a scheduled one, and covers no window', async () => {
    // It was handed a chosen set of traces rather than a stretch of time, so there is no window
    // for the column to name.
    storedRuns = [adhoc()];
    renderPage();

    const row = await rowFor('openai:/gpt-5.6-sol');
    expect(await within(row).findByText('Ad hoc')).toBeInTheDocument();
    expect(within(row).queryByText('Scheduled')).not.toBeInTheDocument();
    expect(row).not.toHaveTextContent(/Jan 1, 12:00/);
    expect(within(row).getByText('40')).toBeInTheDocument();
  });

  test('lists both kinds together, newest first as the search returned them', async () => {
    storedRuns = [adhoc(), scheduled()];
    renderPage();

    expect(await within(await rowFor('openai:/gpt-5.6-sol')).findByText('Ad hoc')).toBeInTheDocument();
    expect(await within(await rowFor('openai:/gpt-5-mini')).findByText('Scheduled')).toBeInTheDocument();
  });

  test('a run that analyzed nothing reads as finished over zero traces', async () => {
    // Rather than a word of this page's own for it: "Finished" beside a scanned count of 0 says
    // the window held nothing it could use, and keeps the status column to the run's own status.
    storedRuns = [
      scheduled({
        runUuid: 'run-quiet',
        tags: { total_traces_analyzed: '0', total_cost_usd: '0.0', model: 'openai:/gpt-5-mini' },
      }),
    ];
    renderPage();

    const row = await rowFor('openai:/gpt-5-mini');
    expect(await within(row).findByText('Finished')).toBeInTheDocument();
    // Scanned traces and new issues both read 0, which together say what happened: it ran, and
    // there was nothing in the window for it to look at.
    await waitFor(() => expect(within(row).getAllByText('0')).toHaveLength(2));
  });

  test('a run from before the analyzed count was recorded claims no count', async () => {
    // The submitted count it does carry is a different number -- the sampler caps what is
    // analyzed below it -- so reporting that one would overstate what the run looked at.
    storedRuns = [scheduled({ runUuid: 'run-old', tags: { total_traces: '640', model: 'openai:/gpt-5-mini' } })];
    renderPage();

    const row = await rowFor('openai:/gpt-5-mini');
    expect(await within(row).findByText('Finished')).toBeInTheDocument();
    expect(within(row).queryByText('640')).not.toBeInTheDocument();
  });

  test('a cancelled run is told apart from one that broke', async () => {
    storedRuns = [scheduled({ runUuid: 'run-killed', status: 'KILLED' })];
    renderPage();

    const row = await rowFor('openai:/gpt-5-mini');
    expect(await within(row).findByText('Killed')).toBeInTheDocument();
    expect(within(row).queryByText('Failed')).not.toBeInTheDocument();
  });

  test('a status this build does not know is named rather than left blank', async () => {
    // A newer server can introduce one; the row still has to say something about it.
    storedRuns = [scheduled({ runUuid: 'run-new', status: 'SOMETHING_NEW' })];
    renderPage();

    expect(await within(await rowFor('openai:/gpt-5-mini')).findByText('Unknown')).toBeInTheDocument();
  });

  test('a failed run reports as failed and claims no cost', async () => {
    storedRuns = [
      scheduled({
        runUuid: 'run-bad',
        status: 'FAILED',
        tags: {
          total_traces_analyzed: '184',
          model: 'openai:/gpt-5-mini',
          [MLFLOW_ISSUE_DETECTION_ERROR_TAG]: 'MlflowException: no API key available',
        },
      }),
    ];
    renderPage();

    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    const row = await rowFor('openai:/gpt-5-mini');
    expect(await within(row).findByText('Failed')).toBeInTheDocument();
    expect(within(row).queryByText(/^\$/)).not.toBeInTheDocument();
    // Not on the page until it is asked for: a row of its own pushed the next run down the page,
    // and a message that runs to a few hundred characters cannot be read in a cell.
    expect(screen.queryByText('MlflowException: no API key available')).not.toBeInTheDocument();

    // The provider's own wording, which is the part worth acting on. Without it the row says only
    // that something broke, and the exception reaches nowhere an experiment's owner can read it.
    await user.hover(within(row).getByText('Failed'));
    expect(await screen.findByRole('tooltip')).toHaveTextContent('MlflowException: no API key available');
  });

  test('a run that failed before the reason was recorded shows no empty detail row', async () => {
    storedRuns = [
      scheduled({
        runUuid: 'run-bad',
        status: 'FAILED',
        tags: { total_traces_analyzed: '1', model: 'openai:/gpt-5-mini' },
      }),
    ];
    renderPage();

    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    const row = await rowFor('openai:/gpt-5-mini');
    expect(await within(row).findByText('Failed')).toBeInTheDocument();

    // Nothing to show, so hovering the status offers nothing rather than an empty popup. The
    // tooltip opens in 10ms under test, so a fifth of a second is long enough to be sure.
    await user.hover(within(row).getByText('Failed'));
    await expect(screen.findByRole('tooltip', {}, { timeout: 200 })).rejects.toThrow();
  });

  test('a run still going claims no issue count yet', async () => {
    // Its count so far would read as a result; it has not finished filing.
    issueSourceRunIds = [];
    storedRuns = [scheduled({ runUuid: 'run-live', status: 'RUNNING', endTime: undefined })];
    renderPage();

    const row = await rowFor('openai:/gpt-5-mini');
    await waitFor(() => expect(within(row).getByText('Running')).toBeInTheDocument());
    expect(within(row).queryByText('0')).not.toBeInTheDocument();
  });

  test('a run still going shows no cost, and is timed against now', async () => {
    storedRuns = [
      scheduled({ runUuid: 'run-live', status: 'RUNNING', startTime: nowMs - 124_000, endTime: undefined }),
    ];
    renderPage();

    const row = await rowFor('openai:/gpt-5-mini');
    expect(await within(row).findByText('Running')).toBeInTheDocument();
    // Measured against now, so a run with no end time does not read as having started in 1970.
    expect(within(row).getByText(/^2m 0[45]s$/)).toBeInTheDocument();
  });

  test('a run abandoned by a dead worker is timed in hours, not in hundreds of minutes', async () => {
    // It stays RUNNING, so it is timed against now. Minutes alone read as "915m 04s" by the next
    // morning, which says less about the run than it does about the formatter.
    storedRuns = [
      scheduled({
        runUuid: 'run-stuck',
        status: 'RUNNING',
        startTime: nowMs - 15 * 60 * 60 * 1000,
        endTime: undefined,
      }),
    ];
    renderPage();

    const row = await rowFor('openai:/gpt-5-mini');
    expect(await within(row).findByText(/^15h 0[01]m$/)).toBeInTheDocument();
  });

  test('an issue count that cannot be read leaves the cell blank rather than reading zero', async () => {
    // Zero would say the run found nothing, which is a different claim from "unknown".
    issueSearchFails = true;
    renderPage();

    const row = await rowFor('openai:/gpt-5-mini');
    await waitFor(() => expect(within(row).getByText('Finished')).toBeInTheDocument());
    expect(within(row).queryByText('0')).not.toBeInTheDocument();
  });

  test('pages forward and back, the way the other list pages do', async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    storedRuns = manyRuns(DETECTION_RUNS_PAGE_SIZE + 2);
    renderPage();

    expect(await screen.findByText(`openai:/m-${DETECTION_RUNS_PAGE_SIZE - 1}`)).toBeInTheDocument();
    expect(screen.queryByText(`openai:/m-${DETECTION_RUNS_PAGE_SIZE}`)).not.toBeInTheDocument();

    await user.click(await screen.findByRole('button', { name: /next/i }));
    expect(await screen.findByText(`openai:/m-${DETECTION_RUNS_PAGE_SIZE + 1}`)).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByText('openai:/m-0')).not.toBeInTheDocument());

    // Back to the cursor it came from, rather than to a fresh first page.
    await user.click(await screen.findByRole('button', { name: /previous/i }));
    expect(await screen.findByText('openai:/m-0')).toBeInTheDocument();
  });

  test('a single page offers no pagination at all', async () => {
    renderPage();

    expect(await screen.findByText('openai:/gpt-5-mini')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /next/i })).not.toBeInTheDocument();
  });

  test('an experiment with no detection runs names when the schedule first fires', async () => {
    storedRuns = [];
    renderPage();

    expect(await screen.findByText('No detection runs yet')).toBeInTheDocument();
    expect(await screen.findByText(/The schedule next runs Jan 18, 2027, 21:15/)).toBeInTheDocument();
  });

  test('an empty page shows the state on its own, with no column headers', async () => {
    // A column header over no rows labels nothing, so it is withheld until there is a row.
    storedRuns = [];
    renderPage();

    expect(await screen.findByText('No detection runs yet')).toBeInTheDocument();
    expect(screen.queryByText('Window covered')).not.toBeInTheDocument();
    expect(screen.queryByText('Analyzed traces')).not.toBeInTheDocument();
    expect(screen.queryByText('Trigger')).not.toBeInTheDocument();
  });

  test('an empty page with no schedule points at the Issues tab instead', async () => {
    // Nothing to promise: without a schedule the only way a run appears here is somebody asking
    // for one, so naming a first run would be naming a run that is not coming.
    experimentTags = [scheduleTag(null)];
    storedRuns = [];
    renderPage();

    expect(
      await screen.findByText(
        'Runs appear here once you detect issues from the Issues tab, or once you turn on a schedule.',
      ),
    ).toBeInTheDocument();
  });

  test('an empty page keeps the schedule line above it', async () => {
    // Nothing duplicated there: the title says no runs yet, the line says what the schedule will do.
    storedRuns = [];
    renderPage();

    expect(await screen.findByText('No detection runs yet')).toBeInTheDocument();
    expect(screen.getByText(/Scheduled weekly on Monday at 21:15 \(UTC\)/)).toBeInTheDocument();
  });

  test('opens the schedule dialog without leaving the page', async () => {
    // The Issues tab's own schedule control, rather than a button of this page's: editing used to
    // mean navigating back to the tab to find the dialog there.
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderPage();

    // Disabled until the schedule has loaded, so the dialog never opens against unknown state.
    const trigger = { name: /scheduled: weekly/i };
    await waitFor(async () => expect(await screen.findByRole('button', trigger)).not.toBeDisabled());
    await user.click(await screen.findByRole('button', trigger));

    expect(await screen.findByRole('button', { name: 'Save' })).toBeInTheDocument();
    expect(screen.getByText('Issue detection run history')).toBeInTheDocument();
    // And the footer's link to this page is withheld here: it would lead where the user is.
    expect(screen.queryByRole('link', { name: 'View run history' })).not.toBeInTheDocument();
  });

  test('a failed read says so instead of claiming nothing has ever run', async () => {
    // Left to the empty state, a transient outage would tell someone this experiment has no
    // history -- the one thing this page exists to answer.
    runSearchFails = true;
    renderPage();

    expect(await screen.findByText("Could not load this experiment's run history")).toBeInTheDocument();
    expect(screen.queryByText('No detection runs yet')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
  });

  test('a page that fails to load leaves the way back to the one before it', async () => {
    // Reporting the failure is right -- the user asked for that page -- but the error state is
    // the whole table, so without the pagination beside it there is nothing left to click and
    // the page they were reading is unreachable.
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    storedRuns = manyRuns(DETECTION_RUNS_PAGE_SIZE + 2);
    renderPage();
    expect(await screen.findByText('openai:/m-0')).toBeInTheDocument();

    runSearchFails = true;
    await user.click(await screen.findByRole('button', { name: /next/i }));
    await waitFor(() => expect(runSearchFailures).toBeGreaterThan(0));
    expect(await screen.findByText("Could not load this experiment's run history")).toBeInTheDocument();

    runSearchFails = false;
    await user.click(await screen.findByRole('button', { name: /previous/i }));
    expect(await screen.findByText('openai:/m-0')).toBeInTheDocument();
  });

  test('a paused schedule names no next run even when its dispatch state still has one', async () => {
    // The state tag keeps the fire time it was last given, and the dispatcher will not honour it
    // while paused, so naming it beside "Schedule is off" would promise a run that never comes.
    experimentTags = [scheduleTag('weekly', { paused: true }), scheduleStateTag(NEXT_FIRE_MS)];
    renderPage();

    expect(await screen.findByText('Schedule is off')).toBeInTheDocument();
    expect(screen.queryByText(/Jan 18, 2027/)).not.toBeInTheDocument();
  });

  test('is not reachable while the Issues tab is off', async () => {
    jest.mocked(shouldEnableExperimentIssuesTab).mockReturnValue(false);
    renderPage();

    expect(await screen.findByText(/Page not found/i)).toBeInTheDocument();
  });
});
