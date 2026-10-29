import { beforeEach, describe, expect, test } from '@jest/globals';
import { rest } from 'msw';
import userEvent, { PointerEventsCheckLevel } from '@testing-library/user-event';

import { MemoryRouter } from '../../../common/utils/RoutingUtils';
import { renderWithDesignSystem, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { setupServer } from '../../../common/utils/setup-msw';
import { QueryClient, QueryClientProvider } from '../../../common/utils/reactQueryHooks';
import {
  MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE_TAG,
  MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG,
  MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER_TAG,
} from '../../constants';
import { ExperimentIssuesScheduleControl } from './ExperimentIssuesScheduleControl';
import { ISSUE_DETECTION_SCHEDULE_VERSION, MAX_TRACE_FILTERS, scheduleTiming } from './issueDetectionSchedule';

interface SavedTag {
  key: string;
  value: string;
}

let experimentTags: SavedTag[];
let written: SavedTag[];
let saveFails: boolean;
/** Key of a tag whose write should fail, to exercise a partial save. */
let failWriteOfTag: string | null;
/** Set to fail the experiment read, which is how a load failure is exercised. */
let loadFails: boolean;
/** How many times the experiment (and so the schedule) has been read. */
let experimentReads: number;
/** How many times a save asked the server to bring the dispatch state up to date. */
let scheduleStateSyncs: number;
/** What that call answers with, and whether it fails at all. */
let syncedNextRunMs: number | null;
let syncFails: boolean;
let scheduledDetectionEnabled: boolean;
let gatewayEndpoints: { name: string }[];
let gatewaySecrets: { secret_id: string; secret_name: string; provider: string }[];
/** What the server reports about a credential of its own for the selected provider. */
let envCredentialsAvailable: boolean | undefined;
let createdSecrets: { secret_name: string; provider?: string; secret_value: Record<string, string> }[];

const server = setupServer(
  rest.get('*/ajax-api/3.0/mlflow/server-info', (_req, res, ctx) =>
    res(
      ctx.json({
        store_type: 'SqlStore',
        features_enabled: { gateway: true, scheduled_issue_detection: scheduledDetectionEnabled },
      }),
    ),
  ),
  rest.get('*/ajax-api/2.0/mlflow/experiments/get', (_req, res, ctx) => {
    experimentReads += 1;
    return loadFails
      ? res(ctx.status(500), ctx.json({ message: 'nope' }))
      : res(ctx.json({ experiment: { experiment_id: '1', name: 'exp', tags: experimentTags } }));
  }),
  rest.post('*/ajax-api/2.0/mlflow/experiments/set-experiment-tag', async (req, res, ctx) => {
    if (saveFails) {
      return res(ctx.status(500), ctx.json({ message: 'nope' }));
    }
    const body = (await req.json()) as SavedTag;
    if (failWriteOfTag === body.key) {
      return res(ctx.status(500), ctx.json({ message: 'nope' }));
    }
    written.push({ key: body.key, value: body.value });
    experimentTags = [...experimentTags.filter(({ key }) => key !== body.key), { key: body.key, value: body.value }];
    return res(ctx.json({}));
  }),
  rest.get('*/ajax-api/3.0/mlflow/gateway/endpoints/list', (_req, res, ctx) =>
    res(ctx.json({ endpoints: gatewayEndpoints })),
  ),
  rest.get('*/ajax-api/3.0/mlflow/gateway/secrets/list', (_req, res, ctx) =>
    res(ctx.json({ secrets: gatewaySecrets })),
  ),
  rest.get('*/ajax-api/3.0/mlflow/gateway/provider-config', (_req, res, ctx) =>
    res(
      ctx.json({
        auth_modes: [],
        default_mode: 'api_key',
        ...(envCredentialsAvailable === undefined ? {} : { env_credentials_available: envCredentialsAvailable }),
      }),
    ),
  ),
  rest.post('*/ajax-api/3.0/mlflow/issues/schedule/state', (_req, res, ctx) => {
    scheduleStateSyncs += 1;
    if (syncFails) {
      return res(ctx.status(500), ctx.json({ message: 'nope' }));
    }
    // What the server does: it writes the dispatch state against the schedule now stored, and
    // only then answers. Stubbing the reply alone would have the next read find no state tag and
    // drop the line again, which is not what a real save leaves behind.
    const saved = JSON.parse(written.findLast(({ key }) => key === MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG)?.value ?? '{}');
    experimentTags = [
      ...experimentTags.filter(({ key }) => key !== MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE_TAG),
      {
        key: MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE_TAG,
        value: JSON.stringify({
          version: ISSUE_DETECTION_SCHEDULE_VERSION,
          lastFireTimeMs: 1,
          timing: scheduleTiming(saved.cadence, saved.fireTime, saved.weekday, saved.timezone),
          nextFireTimeMs: syncedNextRunMs,
        }),
      },
    ];
    return res(ctx.json({ next_fire_time_ms: syncedNextRunMs }));
  }),
  rest.post('*/ajax-api/3.0/mlflow/gateway/secrets/create', async (req, res, ctx) => {
    const body = (await req.json()) as (typeof createdSecrets)[number];
    createdSecrets.push(body);
    const secret = { secret_id: 'sec-new', secret_name: body.secret_name, provider: body.provider ?? '' };
    gatewaySecrets = [...gatewaySecrets, secret];
    return res(ctx.json({ secret }));
  }),
);

/** `fireTime` is left out when not given, which is how a schedule saved before it is exercised. */
const scheduleTag = (
  cadence: string | null,
  paused = false,
  timezone = 'UTC',
  fireTime?: unknown,
  weekday?: unknown,
): SavedTag => ({
  key: MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG,
  value: JSON.stringify({
    version: ISSUE_DETECTION_SCHEDULE_VERSION,
    cadence,
    timezone,
    paused,
    ...(fireTime === undefined ? {} : { fireTime }),
    ...(weekday === undefined ? {} : { weekday }),
  }),
});

const traceFilterTag = (filters: unknown[]): SavedTag => ({
  key: MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER_TAG,
  value: JSON.stringify(filters),
});

/**
 * A router is required because the dialog's footer links out to the run history page, and `at` is
 * the route it is opened from, which decides whether that link leads anywhere.
 */
const renderAt = (at = '/experiments/1/issues') => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, cacheTime: 0 } } });
  return renderWithDesignSystem(
    <MemoryRouter initialEntries={[at]}>
      <QueryClientProvider client={client}>
        <ExperimentIssuesScheduleControl experimentId="1" />
      </QueryClientProvider>
    </MemoryRouter>,
  );
};

const render = () => renderAt();

const setup = () => userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

const COMPONENT_ID = 'mlflow.experiment-issues.schedule';

const savedSchedule = (): Record<string, unknown> =>
  JSON.parse(written.find(({ key }) => key === MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG)?.value ?? '{}');

const savedTraceFilters = (): unknown[] =>
  JSON.parse(written.find(({ key }) => key === MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER_TAG)?.value ?? 'null');

const openModal = async (user: ReturnType<typeof setup>, name: RegExp | string = /schedule detection/i) => {
  // Re-queried inside the wait rather than captured once: until `server-info` answers, the control
  // renders its disabled branch, and becoming available swaps in a different button. Waiting on the
  // element found first would wait forever on a node that has been detached.
  await waitFor(async () => expect(await screen.findByRole('button', { name })).not.toBeDisabled());
  await user.click(await screen.findByRole('button', { name }));
  return screen.findByRole('dialog');
};

const chooseCadence = async (user: ReturnType<typeof setup>, cadence: string) => {
  await user.click(await screen.findByRole('combobox', { name: /cadence/i }));
  await user.click(await screen.findByRole('option', { name: cadence }));
};

const hourSelect = () => screen.findByRole('combobox', { name: 'Hour' });
const minuteSelect = () => screen.findByRole('combobox', { name: 'Minute' });
const weekdaySelect = () => screen.findByRole('combobox', { name: 'Day' });

const chooseOption = async (user: ReturnType<typeof setup>, select: HTMLElement, option: string) => {
  await user.click(select);
  await user.click(await screen.findByRole('option', { name: option }));
};

const chooseStartTime = async (user: ReturnType<typeof setup>, hour: string, minute: string) => {
  await chooseOption(user, await hourSelect(), hour);
  await chooseOption(user, await minuteSelect(), minute);
};

const save = async (user: ReturnType<typeof setup>) => user.click(await screen.findByRole('button', { name: 'Save' }));

describe('ExperimentIssuesScheduleControl', () => {
  beforeEach(() => {
    experimentTags = [];
    written = [];
    saveFails = false;
    failWriteOfTag = null;
    loadFails = false;
    experimentReads = 0;
    scheduleStateSyncs = 0;
    syncedNextRunMs = null;
    syncFails = false;
    scheduledDetectionEnabled = true;
    gatewayEndpoints = [];
    gatewaySecrets = [];
    createdSecrets = [];
    // The default server here already has a credential of its own, which is what keeps the
    // schedule tests about cadence and filters rather than about keys.
    envCredentialsAvailable = true;
  });

  test('labels the trigger "Schedule detection" when off and shows the cadence when scheduled', async () => {
    const { unmount } = render();
    expect(await screen.findByRole('button', { name: /schedule detection/i })).toBeInTheDocument();
    unmount();

    experimentTags = [scheduleTag('hourly')];
    render();
    expect(await screen.findByRole('button', { name: /scheduled: hourly/i })).toBeInTheDocument();
  });

  test("swaps the trigger's clock for a green checkmark while a schedule is on", async () => {
    const { unmount } = render();
    await screen.findByRole('button', { name: /schedule detection/i });
    expect(screen.queryByTestId('schedule-on-icon')).not.toBeInTheDocument();
    unmount();

    experimentTags = [scheduleTag('daily')];
    render();
    const trigger = await screen.findByRole('button', { name: /scheduled: daily/i });
    // In the label, not the leading icon slot, whose colour the button sets.
    expect(trigger).toContainElement(screen.getByTestId('schedule-on-icon'));
  });

  test('a paused schedule reads as Off while keeping the cadence it had', async () => {
    experimentTags = [scheduleTag('weekly', true)];
    const user = setup();
    render();

    await openModal(user);
    expect(await screen.findByRole('combobox', { name: /cadence/i })).toHaveTextContent('Off');

    // Turning it back on offers the cadence it was paused at, rather than starting from nothing.
    await chooseCadence(user, 'Weekly');
    await save(user);
    await waitFor(() => expect(savedSchedule()['cadence']).toBe('weekly'));
    expect(savedSchedule()['paused']).toBe(false);
  });

  test('closes the modal from the footer without saving', async () => {
    const user = setup();
    render();
    await openModal(user);

    await user.click(await screen.findByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    expect(written).toEqual([]);
  });

  test('shows only the Add filter button when there are no clauses, and saves none', async () => {
    const user = setup();
    render();
    await openModal(user);

    expect(screen.queryByRole('combobox', { name: 'Field' })).not.toBeInTheDocument();
    await chooseCadence(user, 'Daily');
    await save(user);

    await waitFor(() => expect(savedTraceFilters()).toEqual([]));
    expect(savedSchedule()['cadence']).toBe('daily');
    // Saved in the browser's zone, so the cadence keeps firing at the local hour it was set in.
    expect(savedSchedule()['timezone']).toEqual(expect.any(String));
  });

  test('adds and saves clauses, defaulting a new row to the metadata field and the equals operator', async () => {
    const user = setup();
    render();
    await openModal(user);
    await chooseCadence(user, 'Daily');

    await user.click(await screen.findByRole('button', { name: 'Add filter' }));
    await user.type(await screen.findByRole('textbox', { name: 'Filter key' }), 'environment');
    await user.type(await screen.findByRole('textbox', { name: 'Filter value' }), 'production');
    await save(user);

    await waitFor(() =>
      expect(savedTraceFilters()).toEqual([
        { field: 'metadata', operator: '=', key: 'environment', value: 'production' },
      ]),
    );
  });

  test('prefills mixed metadata and tag clauses and preserves field and operator on save', async () => {
    const stored = [
      { field: 'metadata', operator: '=', key: 'environment', value: 'production' },
      { field: 'tag', operator: '=', key: 'team', value: 'search' },
    ];
    experimentTags = [scheduleTag('daily'), traceFilterTag(stored)];
    const user = setup();
    render();

    await openModal(user, /scheduled: daily/i);
    const keys = await screen.findAllByRole('textbox', { name: 'Filter key' });
    expect(keys.map((input) => (input as HTMLInputElement).value)).toEqual(['environment', 'team']);

    await save(user);
    await waitFor(() => expect(savedTraceFilters()).toEqual(stored));
  });

  test('drops a removed clause from the saved model', async () => {
    experimentTags = [
      scheduleTag('daily'),
      traceFilterTag([
        { field: 'metadata', operator: '=', key: 'environment', value: 'production' },
        { field: 'tag', operator: '=', key: 'team', value: 'search' },
      ]),
    ];
    const user = setup();
    render();

    await openModal(user, /scheduled: daily/i);
    const removeButtons = await screen.findAllByRole('button', { name: 'Remove filter' });
    await user.click(removeButtons[0]);
    await save(user);

    await waitFor(() =>
      expect(savedTraceFilters()).toEqual([{ field: 'tag', operator: '=', key: 'team', value: 'search' }]),
    );
  });

  test('removing the last clause collapses back to just the Add filter button', async () => {
    experimentTags = [
      scheduleTag('daily'),
      traceFilterTag([{ field: 'tag', operator: '=', key: 'team', value: 'search' }]),
    ];
    const user = setup();
    render();

    await openModal(user, /scheduled: daily/i);
    await user.click(await screen.findByRole('button', { name: 'Remove filter' }));
    expect(screen.queryByRole('textbox', { name: 'Filter key' })).not.toBeInTheDocument();
    expect(await screen.findByRole('button', { name: 'Add filter' })).toBeInTheDocument();
  });

  test('a blank-key clause is dropped rather than saved', async () => {
    const user = setup();
    render();
    await openModal(user);
    await chooseCadence(user, 'Daily');

    await user.click(await screen.findByRole('button', { name: 'Add filter' }));
    await user.type(await screen.findByRole('textbox', { name: 'Filter value' }), 'production');
    await save(user);

    await waitFor(() => expect(savedTraceFilters()).toEqual([]));
  });

  test('holds Save until the model selection resolves, so an active schedule never saves without one', async () => {
    // The endpoint list is what the default selection is derived from, and the credential queries
    // can settle before it. Saving in that gap stored no model -- the run would fall back to the
    // server default rather than the endpoint this dialog was about to show -- and skipped the key
    // step, since that only applies to a direct provider selection.
    envCredentialsAvailable = false;
    gatewayEndpoints = [{ name: 'prod-gateway' }];
    let releaseEndpoints: () => void = () => {};
    const endpointsHeld = new Promise<void>((resolve) => {
      releaseEndpoints = resolve;
    });
    server.use(
      rest.get('*/ajax-api/3.0/mlflow/gateway/endpoints/list', async (_req, res, ctx) => {
        await endpointsHeld;
        return res(ctx.json({ endpoints: gatewayEndpoints }));
      }),
    );

    const user = setup();
    render();
    await openModal(user);
    await chooseCadence(user, 'Daily');

    // The credential answers have arrived; the endpoint list has not.
    await waitFor(() => expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled());
    expect(written).toEqual([]);

    releaseEndpoints();
    await waitFor(() => expect(screen.getByRole('button', { name: 'Save' })).not.toBeDisabled());
    await save(user);

    // The resolved endpoint is stored rather than nothing, and it needs no API key of its own.
    await waitFor(() => expect(savedSchedule()['cadence']).toBe('daily'));
    expect(savedSchedule()['model']).toEqual({ endpointName: 'prod-gateway' });
    expect(screen.queryByTestId('schedule-api-key-view')).not.toBeInTheDocument();
  });

  test('prefills the stored start time and saves the one chosen instead', async () => {
    experimentTags = [scheduleTag('daily', false, 'UTC', '06:45')];
    const user = setup();
    render();

    await openModal(user, /scheduled: daily/i);
    expect(await hourSelect()).toHaveTextContent('06');
    expect(await minuteSelect()).toHaveTextContent('45');

    await chooseStartTime(user, '21', '30');
    await save(user);
    await waitFor(() => expect(savedSchedule()['fireTime']).toBe('21:30'));
  });

  test('a schedule storing no start time keeps firing at the hour it already did', async () => {
    // The compatibility property this control has to hold: opening and saving a schedule from
    // before the start time existed must not move when it fires.
    experimentTags = [scheduleTag('daily')];
    const user = setup();
    render();

    await openModal(user, /scheduled: daily/i);
    expect(await hourSelect()).toHaveTextContent('09');
    expect(await minuteSelect()).toHaveTextContent('00');

    await save(user);
    await waitFor(() => expect(savedSchedule()['fireTime']).toBe('09:00'));
  });

  test('offers the weekday only for a weekly cadence, and saves the one chosen', async () => {
    // Weekly is the only cadence with a day to choose; a daily one fires every day, so offering a
    // day there would be a control with nothing to say.
    experimentTags = [scheduleTag('weekly', false, 'UTC', '17:45', 3)];
    const user = setup();
    render();

    await openModal(user, /scheduled: weekly/i);
    expect(await weekdaySelect()).toHaveTextContent('Thursday');

    await chooseOption(user, await weekdaySelect(), 'Sunday');
    await save(user);
    await waitFor(() => expect(savedSchedule()['weekday']).toBe(6));
    expect(savedSchedule()['fireTime']).toBe('17:45');
  });

  test('drops the weekday when the cadence stops being weekly', async () => {
    experimentTags = [scheduleTag('weekly', false, 'UTC', '17:45', 3)];
    const user = setup();
    render();

    await openModal(user, /scheduled: weekly/i);
    expect(await weekdaySelect()).toBeInTheDocument();

    await chooseCadence(user, 'Daily');
    await waitFor(() => expect(screen.queryByRole('combobox', { name: 'Day' })).not.toBeInTheDocument());

    // Still stored, so switching back to weekly restores the day rather than resetting it.
    await save(user);
    await waitFor(() => expect(savedSchedule()['cadence']).toBe('daily'));
    expect(savedSchedule()['weekday']).toBe(3);
  });

  test('shows the stored timezone with its offset, and saves the one chosen instead', async () => {
    // A time of day says nothing without its zone: an owner editing a Tokyo schedule from
    // anywhere else would otherwise read 17:45 as their own 17:45.
    experimentTags = [scheduleTag('daily', false, 'Asia/Tokyo', '17:45')];
    const user = setup();
    render();

    await openModal(user, /scheduled: daily/i);
    const timezoneInput = await screen.findByPlaceholderText('Search timezones');
    expect(timezoneInput).toHaveValue('(UTC+09:00) Asia/Tokyo');

    await user.click(timezoneInput);
    await user.keyboard('Berlin');
    // By test id rather than by label: the label carries the zone's *current* offset, so matching
    // on it would make this test fail twice a year when Berlin changes to summer time.
    await user.click(await screen.findByTestId(`${COMPONENT_ID}.modal.timezone.option.Europe/Berlin`));
    await save(user);

    await waitFor(() => expect(savedSchedule()['timezone']).toBe('Europe/Berlin'));
    // The wall-clock time is unchanged -- changing the zone moves the instant, not the time of day.
    expect(savedSchedule()['fireTime']).toBe('17:45');
  });

  test('offers a stored start time that is not one of the quarter hours, and keeps it', async () => {
    // The dispatcher polls every 15 minutes, so 09:08 fires at 09:15 -- late, but it does fire.
    // Rounding it here would change when a schedule runs because someone opened the dialog.
    experimentTags = [scheduleTag('daily', false, 'UTC', '09:08')];
    const user = setup();
    render();

    await openModal(user, /scheduled: daily/i);
    expect(await hourSelect()).toHaveTextContent('09');
    expect(await minuteSelect()).toHaveTextContent('08');

    await save(user);
    await waitFor(() => expect(savedSchedule()['fireTime']).toBe('09:08'));
  });

  test('falls back to the default rather than blanking the control on an unreadable start time', async () => {
    experimentTags = [scheduleTag('daily', false, 'UTC', 'at noon')];
    const user = setup();
    render();

    await openModal(user, /scheduled: daily/i);
    expect(await hourSelect()).toHaveTextContent('09');
    expect(await minuteSelect()).toHaveTextContent('00');
  });

  test("shows when the saved schedule next runs, in the schedule's own zone", async () => {
    // The question the controls cannot answer on their own: "daily at 16:00" does not say
    // whether that means today or tomorrow.
    const nextRun = Date.UTC(2026, 10, 3, 15, 0);
    experimentTags = [
      scheduleTag('daily', false, 'Europe/Berlin', '16:00', 0),
      {
        key: MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE_TAG,
        value: JSON.stringify({
          version: ISSUE_DETECTION_SCHEDULE_VERSION,
          lastFireTimeMs: nextRun - 86_400_000,
          timing: scheduleTiming('daily', '16:00', 0, 'Europe/Berlin'),
          nextFireTimeMs: nextRun,
        }),
      },
    ];
    const user = setup();
    render();

    await openModal(user, /scheduled: daily/i);
    // 15:00 UTC is 16:00 in Berlin, which is the zone the row above is read in.
    expect(await screen.findByText(/Next run as saved:.*16:00/)).toBeInTheDocument();
  });

  test('says nothing about the next run when the schedule has been retimed since', async () => {
    // The server has not recomputed it yet, and the old value names a time nothing will happen
    // at, so the line is dropped rather than left to mislead.
    experimentTags = [
      scheduleTag('daily', false, 'Europe/Berlin', '16:00', 0),
      {
        key: MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE_TAG,
        value: JSON.stringify({
          version: ISSUE_DETECTION_SCHEDULE_VERSION,
          lastFireTimeMs: 1,
          timing: scheduleTiming('daily', '09:00', 0, 'Europe/Berlin'),
          nextFireTimeMs: Date.UTC(2026, 10, 3, 8, 0),
        }),
      },
    ];
    const user = setup();
    render();

    await openModal(user, /scheduled: daily/i);
    expect(await screen.findByRole('combobox', { name: 'Hour' })).toHaveTextContent('16');
    expect(screen.queryByText(/Next run as saved:/)).not.toBeInTheDocument();
  });

  test('says nothing about the next run while the schedule is off', async () => {
    experimentTags = [
      scheduleTag('daily', true, 'Europe/Berlin', '16:00', 0),
      {
        key: MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE_TAG,
        value: JSON.stringify({
          version: ISSUE_DETECTION_SCHEDULE_VERSION,
          lastFireTimeMs: 1,
          timing: scheduleTiming('daily', '16:00', 0, 'Europe/Berlin'),
          nextFireTimeMs: Date.UTC(2026, 10, 3, 15, 0),
        }),
      },
    ];
    const user = setup();
    render();

    await openModal(user);
    expect(await screen.findByRole('combobox', { name: /cadence/i })).toHaveTextContent('Off');
    expect(screen.queryByText(/Next run as saved:/)).not.toBeInTheDocument();
  });

  test('shows the next run as soon as the schedule is saved', async () => {
    // The tags are written through the generic tag endpoint, so the save asks the server to
    // bring the dispatch state up to date and answer with the next run -- otherwise there is
    // nothing to show until the dispatcher's next pass, a quarter of an hour of silence at the
    // one moment the owner wants to be told what they just scheduled.
    syncedNextRunMs = Date.UTC(2026, 10, 3, 15, 0);
    experimentTags = [scheduleTag('daily', false, 'Europe/Berlin', '09:00', 0)];
    const user = setup();
    render();

    await openModal(user, /scheduled: daily/i);
    await chooseStartTime(user, '16', '00');
    await save(user);

    await waitFor(() => expect(savedSchedule()['fireTime']).toBe('16:00'));
    expect(scheduleStateSyncs).toBe(1);
    await openModal(user, /scheduled: daily/i);
    expect(await screen.findByText(/Next run as saved:.*16:00/)).toBeInTheDocument();
  });

  test('a save still succeeds when the state cannot be brought up to date', async () => {
    // An older server has no such route and an authorization refusal looks the same. The
    // schedule is already written by then, so this must not fail the save -- the dispatcher's
    // next pass computes the next run, which is where it came from before.
    syncFails = true;
    const user = setup();
    render();

    await openModal(user);
    await chooseCadence(user, 'Daily');
    await save(user);

    await waitFor(() => expect(savedSchedule()['cadence']).toBe('daily'));
    expect(await screen.findByRole('button', { name: /scheduled: daily/i })).toBeInTheDocument();
    expect(screen.queryByText(/Couldn't save the schedule/)).not.toBeInTheDocument();
  });

  test('reads the schedule again on open, so a saved next run appears without a reload', async () => {
    // Only the dispatcher computes a next run, so a save resolves with none. Without this the
    // line stayed missing for the rest of the session: the query does not poll and does not
    // refetch on focus.
    const nextRun = Date.UTC(2026, 10, 3, 15, 0);
    experimentTags = [scheduleTag('daily', false, 'Europe/Berlin', '16:00', 0)];
    const user = setup();
    render();

    await openModal(user, /scheduled: daily/i);
    expect(screen.queryByText(/Next run as saved:/)).not.toBeInTheDocument();
    const readsOnFirstOpen = experimentReads;

    await user.click(await screen.findByRole('button', { name: 'Cancel' }));
    // The dispatcher's pass lands between the two opens.
    experimentTags = [
      scheduleTag('daily', false, 'Europe/Berlin', '16:00', 0),
      {
        key: MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE_TAG,
        value: JSON.stringify({
          version: ISSUE_DETECTION_SCHEDULE_VERSION,
          lastFireTimeMs: nextRun - 86_400_000,
          timing: scheduleTiming('daily', '16:00', 0, 'Europe/Berlin'),
          nextFireTimeMs: nextRun,
        }),
      },
    ];

    await openModal(user, /scheduled: daily/i);
    expect(await screen.findByText(/Next run as saved:.*16:00/)).toBeInTheDocument();
    expect(experimentReads).toBeGreaterThan(readsOnFirstOpen);
  });

  test('keeps naming the saved next run while the draft is edited', async () => {
    // Only the server computes these, and it has not seen an unsaved edit -- so the value must
    // not follow the draft, and the label has to say which schedule it is describing. Pinned
    // because the alternative is a line that silently answers for the controls above it.
    const nextRun = Date.UTC(2026, 10, 3, 15, 0);
    experimentTags = [
      scheduleTag('daily', false, 'Europe/Berlin', '16:00', 0),
      {
        key: MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE_TAG,
        value: JSON.stringify({
          version: ISSUE_DETECTION_SCHEDULE_VERSION,
          lastFireTimeMs: nextRun - 86_400_000,
          timing: scheduleTiming('daily', '16:00', 0, 'Europe/Berlin'),
          nextFireTimeMs: nextRun,
        }),
      },
    ];
    const user = setup();
    render();

    await openModal(user, /scheduled: daily/i);
    expect(await screen.findByText(/Next run as saved:.*16:00/)).toBeInTheDocument();

    await chooseStartTime(user, '07', '45');
    await chooseCadence(user, 'Weekly');

    expect(await screen.findByRole('combobox', { name: 'Hour' })).toHaveTextContent('07');
    // Still the saved schedule's next run, and still labelled as such.
    expect(await screen.findByText(/Next run as saved:.*16:00/)).toBeInTheDocument();
  });

  test('drops the next run as soon as the draft is turned off', async () => {
    // The line describes the saved schedule, but a draft set to Off is about to stop it, so
    // leaving it on screen would describe a schedule the user has just turned off.
    const nextRun = Date.UTC(2026, 10, 3, 15, 0);
    experimentTags = [
      scheduleTag('daily', false, 'Europe/Berlin', '16:00', 0),
      {
        key: MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE_TAG,
        value: JSON.stringify({
          version: ISSUE_DETECTION_SCHEDULE_VERSION,
          lastFireTimeMs: nextRun - 86_400_000,
          timing: scheduleTiming('daily', '16:00', 0, 'Europe/Berlin'),
          nextFireTimeMs: nextRun,
        }),
      },
    ];
    const user = setup();
    render();

    await openModal(user, /scheduled: daily/i);
    expect(await screen.findByText(/Next run as saved:/)).toBeInTheDocument();

    await chooseCadence(user, 'Off');
    await waitFor(() => expect(screen.queryByText(/Next run as saved:/)).not.toBeInTheDocument());
  });

  test('says the cost recurs while a cadence is selected, and not when off', async () => {
    const costNote = /incur model usage costs on every occurrence/;
    experimentTags = [scheduleTag('hourly')];
    const user = setup();
    render();

    await openModal(user, /scheduled: hourly/i);
    expect(await screen.findByText(costNote)).toBeInTheDocument();

    // Turning the schedule off stops it firing, so there is no recurring spend to warn about.
    await chooseCadence(user, 'Off');
    await waitFor(() => expect(screen.queryByText(costNote)).not.toBeInTheDocument());
  });

  test('offers no more filters than scheduled runs apply', async () => {
    const user = setup();
    render();
    await openModal(user);
    await chooseCadence(user, 'Daily');

    const addFilter = await screen.findByRole('button', { name: 'Add filter' });
    for (let i = 0; i < MAX_TRACE_FILTERS; i++) {
      await user.click(addFilter);
    }
    expect(await screen.findAllByRole('textbox', { name: 'Filter key' })).toHaveLength(MAX_TRACE_FILTERS);
    // A further clause would show here as active and be dropped by every run.
    expect(addFilter).toBeDisabled();
  });

  test("keeps a schedule's stored timezone when it is edited from somewhere else", async () => {
    // The modal has no timezone control, so a save made elsewhere must not move when the schedule
    // fires: pausing a Tokyo daily schedule from another zone would otherwise resume it at 09:00
    // local to whoever paused it.
    experimentTags = [scheduleTag('daily', false, 'Asia/Tokyo')];
    const user = setup();
    render();

    await openModal(user, /scheduled: daily/i);
    await chooseCadence(user, 'Off');
    await save(user);

    await waitFor(() => expect(savedSchedule()['paused']).toBe(true));
    expect(savedSchedule()['timezone']).toBe('Asia/Tokyo');
  });

  test('adopts the browser timezone only when the schedule has none stored', async () => {
    const user = setup();
    render();
    await openModal(user);
    await chooseCadence(user, 'Daily');
    await save(user);

    await waitFor(() => expect(savedSchedule()['cadence']).toBe('daily'));
    expect(savedSchedule()['timezone']).toBe(Intl.DateTimeFormat().resolvedOptions().timeZone);
  });

  test('turns a schedule off', async () => {
    experimentTags = [scheduleTag('hourly')];
    const user = setup();
    render();

    await openModal(user, /scheduled: hourly/i);
    await chooseCadence(user, 'Off');
    await save(user);

    await waitFor(() => expect(savedSchedule()['paused']).toBe(true));
    // The cadence is kept, so turning it back on restores it.
    expect(savedSchedule()['cadence']).toBe('hourly');
  });

  test('disables the clause inputs and Add filter when the cadence is Off', async () => {
    experimentTags = [
      scheduleTag('hourly'),
      traceFilterTag([{ field: 'tag', operator: '=', key: 'team', value: 'search' }]),
    ];
    const user = setup();
    render();

    await openModal(user, /scheduled: hourly/i);
    expect(await screen.findByRole('textbox', { name: 'Filter key' })).not.toBeDisabled();

    await chooseCadence(user, 'Off');
    expect(await screen.findByRole('textbox', { name: 'Filter key' })).toBeDisabled();
    expect(await screen.findByRole('button', { name: 'Add filter' })).toBeDisabled();
  });

  test('keeps the modal open and shows an error when saving fails', async () => {
    const user = setup();
    render();
    await openModal(user);
    await chooseCadence(user, 'Daily');

    saveFails = true;
    await save(user);

    expect(await screen.findByText(/Couldn't save the schedule/)).toBeInTheDocument();
    // The draft is still there to retry with, rather than discarded by a closing modal.
    expect(await screen.findByRole('dialog')).toBeInTheDocument();
    expect(await screen.findByRole('combobox', { name: /cadence/i })).toHaveTextContent('Daily');
  });

  test('disables the trigger and explains why when the server cannot run a schedule', async () => {
    scheduledDetectionEnabled = false;
    render();

    // Only the unavailable branch wraps the trigger in a tooltip, so that attribute is what says
    // the control took it. Waiting on `toBeDisabled` alone would not discriminate: the ordinary
    // trigger is disabled too while the schedule loads, so that assertion can hold for the wrong
    // reason. The capability itself reads false until `server-info` answers
    // (`useFeatureEnabled(SCHEDULED_ISSUE_DETECTION, false)`), which is the fail-closed direction
    // for a capability question -- a control that looks available and saves a schedule the server
    // cannot run is the worse default.
    await waitFor(async () =>
      expect(await screen.findByRole('button', { name: /schedule detection/i })).toHaveAttribute('tooltip-data-state'),
    );
    expect(await screen.findByRole('button', { name: /schedule detection/i })).toBeDisabled();

    // The reason itself is not asserted here: the design system makes the disabled button its own
    // tooltip trigger, and jsdom delivers no pointer or focus event to a disabled element, so no
    // unit test can open it. Driving it with a simulated hover passed only intermittently, which is
    // worse than not asserting it. The wording is covered by the screenshots on the pull request.
  });

  test('saves the model the picker is left on, defaulting to the first provider', async () => {
    const user = setup();
    render();
    await openModal(user);
    expect(await screen.findByTestId('model-dropdown-trigger')).toHaveTextContent('OpenAI');
    await chooseCadence(user, 'Daily');
    await save(user);

    await waitFor(() => expect(savedSchedule()['model']).toEqual({ provider: 'openai', model: 'gpt-5.6-sol' }));
  });

  test('defaults to the first AI Gateway endpoint when the server has one, as the ad-hoc dialog does', async () => {
    gatewayEndpoints = [{ name: 'prod-gateway' }];
    const user = setup();
    render();
    await openModal(user);
    expect(await screen.findByTestId('model-dropdown-trigger')).toHaveTextContent('prod-gateway');
    await chooseCadence(user, 'Daily');
    await save(user);

    await waitFor(() => expect(savedSchedule()['model']).toEqual({ endpointName: 'prod-gateway' }));
  });

  test('prefills the picker from the stored model', async () => {
    experimentTags = [
      {
        key: MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG,
        value: JSON.stringify({
          version: ISSUE_DETECTION_SCHEDULE_VERSION,
          cadence: 'daily',
          timezone: 'UTC',
          paused: false,
          model: { provider: 'anthropic', model: 'claude-opus-4-8' },
        }),
      },
    ];
    const user = setup();
    render();
    await openModal(user, /scheduled: daily/i);

    const trigger = await screen.findByTestId('model-dropdown-trigger');
    expect(trigger).toHaveTextContent('Anthropic');
    expect(trigger).toHaveTextContent('claude-opus-4-8');
  });

  test('keeps the stored model when the schedule is turned off, so turning it back on restores it', async () => {
    experimentTags = [
      {
        key: MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG,
        value: JSON.stringify({
          version: ISSUE_DETECTION_SCHEDULE_VERSION,
          cadence: 'daily',
          timezone: 'UTC',
          paused: false,
          model: { provider: 'anthropic', model: 'claude-opus-4-8' },
        }),
      },
    ];
    const user = setup();
    render();
    await openModal(user, /scheduled: daily/i);
    await chooseCadence(user, 'Off');
    await save(user);

    await waitFor(() => expect(savedSchedule()['paused']).toBe(true));
    expect(savedSchedule()['model']).toEqual({ provider: 'anthropic', model: 'claude-opus-4-8' });
  });

  test('keeps the credential the schedule is pinned to when an unrelated edit is saved', async () => {
    // The hazard: re-reading the provider's first secret on every save would re-point the
    // schedule at whichever key that provider happens to list first -- on a server with more than
    // one user, somebody else's. Changing the cadence must not touch which credential runs.
    experimentTags = [
      {
        key: MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG,
        value: JSON.stringify({
          version: ISSUE_DETECTION_SCHEDULE_VERSION,
          cadence: 'daily',
          timezone: 'UTC',
          paused: false,
          model: { provider: 'openai', model: 'gpt-5.6-sol', secretId: 'sec-mine' },
        }),
      },
    ];
    gatewaySecrets = [
      { secret_id: 'sec-someone-elses', secret_name: 'bob-openai-key', provider: 'openai' },
      { secret_id: 'sec-mine', secret_name: 'my-openai-key', provider: 'openai' },
    ];
    const user = setup();
    render();
    await openModal(user, /scheduled: daily/i);
    await chooseCadence(user, 'Weekly');
    await save(user);

    await waitFor(() => expect(savedSchedule()['cadence']).toBe('weekly'));
    expect(savedSchedule()['model']).toEqual({
      provider: 'openai',
      model: 'gpt-5.6-sol',
      secretId: 'sec-mine',
    });
  });

  test('keeps the pin when the provider lists no secrets, rather than dropping it', async () => {
    // A read that comes back empty -- or fails -- says nothing about which key the schedule was
    // pinned to. Dropping the pin would send the next occurrence to the server's own credential.
    experimentTags = [
      {
        key: MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG,
        value: JSON.stringify({
          version: ISSUE_DETECTION_SCHEDULE_VERSION,
          cadence: 'daily',
          timezone: 'UTC',
          paused: false,
          model: { provider: 'openai', model: 'gpt-5.6-sol', secretId: 'sec-mine' },
        }),
      },
    ];
    gatewaySecrets = [];
    const user = setup();
    render();
    await openModal(user, /scheduled: daily/i);
    await chooseCadence(user, 'Off');
    await save(user);

    await waitFor(() => expect(savedSchedule()['paused']).toBe(true));
    expect(savedSchedule()['model']).toEqual({
      provider: 'openai',
      model: 'gpt-5.6-sol',
      secretId: 'sec-mine',
    });
  });

  test('rejects a pasted control character, naming it by code point', async () => {
    // The server drops a clause carrying one, so storing it would leave the modal showing a filter
    // that every run ignores -- and a tab shows nothing in the field, hence the code point.
    const user = setup();
    render();
    await openModal(user);
    await chooseCadence(user, 'Daily');
    await user.click(await screen.findByRole('button', { name: 'Add filter' }));
    await user.type(await screen.findByRole('textbox', { name: 'Filter key' }), 'environment');
    await user.click(await screen.findByRole('textbox', { name: 'Filter value' }));
    await user.paste('production\t');
    await save(user);

    expect(await screen.findByText(/U\+0009/)).toBeInTheDocument();
    expect(written).toEqual([]);
  });

  test('asks for an API key before saving when neither the server nor AI Gateway has one', async () => {
    envCredentialsAvailable = false;
    const user = setup();
    render();
    await openModal(user);
    await chooseCadence(user, 'Daily');
    await save(user);

    expect(await screen.findByTestId('schedule-api-key-view')).toBeInTheDocument();
    expect(await screen.findByText(/OpenAI needs an API key/)).toBeInTheDocument();
    // Nothing is stored until the key is in place.
    expect(written).toEqual([]);

    await user.type(await screen.findByTestId('schedule-api-key-input'), 'sk-test-key');
    await user.click(await screen.findByRole('button', { name: /continue and save/i }));

    await waitFor(() => expect(savedSchedule()['cadence']).toBe('daily'));
    expect(createdSecrets).toEqual([
      { secret_name: expect.any(String), provider: 'openai', secret_value: { api_key: 'sk-test-key' } },
    ]);
    // The schedule names the secret just created, so the dispatcher runs on that key and not on
    // whichever one a later read happens to find first.
    expect(savedSchedule()['model']).toEqual({
      provider: 'openai',
      model: 'gpt-5.6-sol',
      secretId: 'sec-new',
    });
  });

  test('names the existing secret when AI Gateway already holds one for the provider', async () => {
    // No key step here -- a key already exists -- but the schedule must still record *which* one,
    // or the dispatcher would be back to guessing among a provider's secrets at fire time.
    gatewaySecrets = [{ secret_id: 'sec-existing', secret_name: 'my-openai-key', provider: 'openai' }];
    const user = setup();
    render();
    await openModal(user);
    await chooseCadence(user, 'Daily');
    await save(user);

    await waitFor(() => expect(savedSchedule()['cadence']).toBe('daily'));
    expect(savedSchedule()['model']).toEqual({
      provider: 'openai',
      model: 'gpt-5.6-sol',
      secretId: 'sec-existing',
    });
  });

  test('stores one key when "Continue and save" is clicked again while the first is in flight', async () => {
    // Each click mints a fresh secret name, so a second one lands a duplicate key for the same
    // schedule. `loading` hides the button's label but still accepts a click, so the guard has to
    // be `disabled`. Reproduced by holding the create response open and clicking again before the
    // first settles -- awaiting the first click instead lets the in-flight state propagate, which
    // hides the race.
    envCredentialsAvailable = false;
    let releaseCreate: () => void = () => {};
    const createHeld = new Promise<void>((resolve) => {
      releaseCreate = resolve;
    });
    server.use(
      rest.post('*/ajax-api/3.0/mlflow/gateway/secrets/create', async (req, res, ctx) => {
        const body = (await req.json()) as (typeof createdSecrets)[number];
        createdSecrets.push(body);
        await createHeld;
        const secret = { secret_id: 'sec-new', secret_name: body.secret_name, provider: body.provider ?? '' };
        gatewaySecrets = [...gatewaySecrets, secret];
        return res(ctx.json({ secret }));
      }),
    );

    const user = setup();
    render();
    await openModal(user);
    await chooseCadence(user, 'Daily');
    await save(user);

    await user.type(await screen.findByTestId('schedule-api-key-input'), 'sk-test-key');
    const continueAndSave = await screen.findByRole('button', { name: /continue and save/i });
    const firstClick = user.click(continueAndSave);
    await waitFor(() => expect(createdSecrets).toHaveLength(1));

    // Found by component id rather than name: the in-flight button hides its label, so its
    // accessible name no longer matches. This is the node a second click would land on.
    const inFlight = () =>
      // eslint-disable-next-line testing-library/no-node-access
      document.querySelector(`[data-component-id="${COMPONENT_ID}.modal.footer.api-key-continue"]`);
    expect(inFlight()).toBeDisabled();
    await user.click(inFlight()!);
    releaseCreate();
    await firstClick;

    await waitFor(() => expect(savedSchedule()['cadence']).toBe('daily'));
    expect(createdSecrets).toHaveLength(1);
  });

  test('reuses the key it already stored when a failed save is retried', async () => {
    // The secret is created before the schedule is written, and the two writes can still fail.
    // Minting a second key on retry would leave an unused credential in AI Gateway per attempt,
    // named randomly, with nothing tying it to this schedule.
    envCredentialsAvailable = false;
    failWriteOfTag = MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG;
    const user = setup();
    render();
    await openModal(user);
    await chooseCadence(user, 'Daily');
    await save(user);

    await user.type(await screen.findByTestId('schedule-api-key-input'), 'sk-demo-key');
    await user.click(await screen.findByRole('button', { name: /continue and save/i }));
    expect(await screen.findByText(/Couldn't save the schedule/)).toBeInTheDocument();
    expect(createdSecrets).toHaveLength(1);

    failWriteOfTag = null;
    await user.click(await screen.findByRole('button', { name: /continue and save/i }));

    await waitFor(() => expect(savedSchedule()['cadence']).toBe('daily'));
    expect(createdSecrets).toHaveLength(1);
    expect(savedSchedule()['model']).toEqual({
      provider: 'openai',
      model: 'gpt-5.6-sol',
      secretId: 'sec-new',
    });
  });

  test('does not ask for a key when AI Gateway already holds one for the provider', async () => {
    envCredentialsAvailable = false;
    gatewaySecrets = [{ secret_id: 'sec-1', secret_name: 'openai-key', provider: 'openai' }];
    const user = setup();
    render();
    await openModal(user);
    await chooseCadence(user, 'Daily');
    await save(user);

    await waitFor(() => expect(savedSchedule()['cadence']).toBe('daily'));
    expect(screen.queryByTestId('schedule-api-key-view')).not.toBeInTheDocument();
  });

  test('does not ask for a key for an AI Gateway endpoint, which carries its own credentials', async () => {
    envCredentialsAvailable = false;
    gatewayEndpoints = [{ name: 'prod-gateway' }];
    const user = setup();
    render();
    await openModal(user);
    await chooseCadence(user, 'Daily');
    await save(user);

    await waitFor(() => expect(savedSchedule()['model']).toEqual({ endpointName: 'prod-gateway' }));
    expect(screen.queryByTestId('schedule-api-key-view')).not.toBeInTheDocument();
  });

  test('turning a schedule off needs no key, since nothing will run', async () => {
    envCredentialsAvailable = false;
    experimentTags = [scheduleTag('daily')];
    const user = setup();
    render();
    await openModal(user, /scheduled: daily/i);
    await chooseCadence(user, 'Off');
    await save(user);

    await waitFor(() => expect(savedSchedule()['paused']).toBe(true));
    expect(screen.queryByTestId('schedule-api-key-view')).not.toBeInTheDocument();
  });

  test('saves without asking for a key when the server does not report whether it has one', async () => {
    // An older server, or one whose gateway cannot be reached: a schedule that could be saved
    // before this step existed must not become unsavable because the check cannot answer.
    envCredentialsAvailable = undefined;
    const user = setup();
    render();
    await openModal(user);
    await chooseCadence(user, 'Daily');
    await save(user);

    await waitFor(() => expect(savedSchedule()['cadence']).toBe('daily'));
    expect(screen.queryByTestId('schedule-api-key-view')).not.toBeInTheDocument();
  });

  test('the API key step can be left without losing the schedule draft', async () => {
    envCredentialsAvailable = false;
    const user = setup();
    render();
    await openModal(user);
    await chooseCadence(user, 'Daily');
    await save(user);

    expect(await screen.findByTestId('schedule-api-key-view')).toBeInTheDocument();
    await user.click(await screen.findByRole('button', { name: 'Back' }));

    expect(await screen.findByRole('combobox', { name: /cadence/i })).toHaveTextContent('Daily');
  });

  test('the filters are stored before the cadence that activates them', async () => {
    const user = setup();
    render();
    await openModal(user);
    await chooseCadence(user, 'Daily');
    await user.click(await screen.findByRole('button', { name: 'Add filter' }));
    await user.type(await screen.findByRole('textbox', { name: 'Filter key' }), 'environment');
    await user.type(await screen.findByRole('textbox', { name: 'Filter value' }), 'production');
    await save(user);

    await waitFor(() => expect(written).toHaveLength(2));
    // The cadence is what makes the dispatcher pick the schedule up, so it must land last: written
    // first, a failed second write or a poll in between could run against clauses not yet stored.
    expect(written.map(({ key }) => key)).toEqual([
      MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER_TAG,
      MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG,
    ]);
  });

  test('a schedule is not activated when its filters cannot be stored', async () => {
    failWriteOfTag = MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER_TAG;
    const user = setup();
    render();
    await openModal(user);
    await chooseCadence(user, 'Daily');
    await save(user);

    expect(await screen.findByText(/Couldn't save the schedule/)).toBeInTheDocument();
    // Nothing was activated, so no run can happen under clauses that were never stored.
    expect(written).toEqual([]);
  });

  test('a clause the server cannot match on is rejected instead of saved', async () => {
    const user = setup();
    render();
    await openModal(user);
    await chooseCadence(user, 'Daily');
    await user.click(await screen.findByRole('button', { name: 'Add filter' }));
    await user.type(await screen.findByRole('textbox', { name: 'Filter key' }), 'team');
    await user.type(await screen.findByRole('textbox', { name: 'Filter value' }), "o'brien");
    await save(user);

    expect(await screen.findByText(/cannot match on it/)).toBeInTheDocument();
    // Nothing is stored, so the modal never shows a filter that every run would ignore.
    expect(written).toEqual([]);

    // Correcting the value clears the message and lets the save through.
    await user.clear(await screen.findByRole('textbox', { name: 'Filter value' }));
    await user.type(await screen.findByRole('textbox', { name: 'Filter value' }), 'obrien');
    await save(user);
    await waitFor(() =>
      expect(savedTraceFilters()).toEqual([{ field: 'metadata', operator: '=', key: 'team', value: 'obrien' }]),
    );
  });

  test('a schedule that could not be read is not editable, and offers a retry', async () => {
    loadFails = true;
    const user = setup();
    render();

    // Opening the modal against a failed read would show "Off" with no filters, and saving that
    // would replace whatever is really stored.
    expect(await screen.findByRole('button', { name: /retry schedule/i })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /schedule detection/i })).not.toBeInTheDocument();

    loadFails = false;
    experimentTags = [scheduleTag('weekly')];
    await user.click(await screen.findByRole('button', { name: /retry schedule/i }));

    expect(await screen.findByRole('button', { name: /scheduled: weekly/i })).toBeInTheDocument();
  });

  describe('run history', () => {
    test('the footer links to the detection run history', async () => {
      experimentTags = [scheduleTag('weekly')];
      const user = setup();
      render();

      await user.click(await screen.findByRole('button', { name: /scheduled: weekly/i }));

      expect(await screen.findByRole('link', { name: 'View run history' })).toHaveAttribute(
        'href',
        '/experiments/1/issues/runs',
      );
    });

    test('the link is hidden on the run history page, where it would lead nowhere', async () => {
      // The same trigger opens this dialog there, so the link would point at the page the user is
      // already reading.
      experimentTags = [scheduleTag('weekly')];
      const user = setup();
      renderAt('/experiments/1/issues/runs');
      await openModal(user, /scheduled: weekly/i);

      expect(await screen.findByRole('button', { name: 'Save' })).toBeInTheDocument();
      expect(screen.queryByRole('link', { name: 'View run history' })).not.toBeInTheDocument();
    });

    test('the link is hidden on the API key step, which is mid-save', async () => {
      // The step is reached by saving, so leaving a navigation link there would offer to walk away
      // from a schedule that has not been stored yet.
      experimentTags = [];
      gatewaySecrets = [];
      envCredentialsAvailable = false;
      const user = setup();
      render();
      await openModal(user);
      await chooseCadence(user, 'Daily');
      await save(user);

      expect(await screen.findByRole('button', { name: 'Continue and save' })).toBeInTheDocument();
      expect(screen.queryByRole('link', { name: 'View run history' })).not.toBeInTheDocument();
    });
  });
});
