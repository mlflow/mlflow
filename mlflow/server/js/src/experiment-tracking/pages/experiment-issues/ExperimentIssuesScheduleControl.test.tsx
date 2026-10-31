import { afterEach, beforeEach, describe, expect, test } from '@jest/globals';
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
/** Fails the experiment read, to exercise a load failure. */
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
/**
 * Saves that have written their trace filters but not yet reached their last call. A save writes the
 * trace filters, then the schedule, then syncs the dispatch state, so a test that stops waiting after
 * an earlier step would let the rest land in the next test's tags.
 */
let savesInFlight: number;

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
      // A failed schedule write ends the save before its sync.
      if (body.key === MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG) {
        savesInFlight -= 1;
      }
      return res(ctx.status(500), ctx.json({ message: 'nope' }));
    }
    if (body.key === MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER_TAG) {
      savesInFlight += 1;
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
    savesInFlight -= 1;
    if (syncFails) {
      return res(ctx.status(500), ctx.json({ message: 'nope' }));
    }
    // The real route writes the state tag against the schedule now stored and only then answers;
    // stubbing the reply alone would leave the next read with no state tag.
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

/** `at` is the route the dialog opens from, which decides whether its history link leads anywhere. */
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
  // Re-queried each poll, because `server-info` answering swaps the control's disabled branch for
  // a different button and detaches the element found first. Synchronously, because a nested
  // `findByRole` polls too and spends this wait's whole budget on one attempt.
  await waitFor(() => expect(screen.getByRole('button', { name })).not.toBeDisabled(), { timeout: 5000 });
  await user.click(screen.getByRole('button', { name }));
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
    savesInFlight = 0;
    syncedNextRunMs = null;
    syncFails = false;
    scheduledDetectionEnabled = true;
    gatewayEndpoints = [];
    gatewaySecrets = [];
    createdSecrets = [];
    // A credential of its own, so these tests stay about cadence and filters rather than keys.
    envCredentialsAvailable = true;
  });

  afterEach(async () => {
    await waitFor(() => {
      // Below zero means a state sync arrived that no save started, so the counting above has fallen
      // behind how a save works; fail on it rather than clamp it away.
      if (savesInFlight !== 0) {
        throw new Error(
          savesInFlight > 0
            ? `${savesInFlight} save(s) still in flight`
            : `${-savesInFlight} state sync(s) arrived that no save started`,
        );
      }
    });
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

    await waitFor(() => expect(savedSchedule()['cadence']).toBe('daily'));
    expect(savedTraceFilters()).toEqual([]);
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
    // The default selection derives from the endpoint list, which can settle after the credential
    // queries. Saving in that gap stored no model, so the run fell back to the server default.
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
    // Opening and saving a schedule from before the start time existed must not move when it fires.
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
    // Weekly is the only cadence with a day to choose; a daily one fires every day.
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
    // A time of day says nothing without its zone: 17:45 in Tokyo is not the reader's 17:45.
    experimentTags = [scheduleTag('daily', false, 'Asia/Tokyo', '17:45')];
    const user = setup();
    render();

    await openModal(user, /scheduled: daily/i);
    const timezoneInput = await screen.findByPlaceholderText('Search timezones');
    expect(timezoneInput).toHaveValue('(UTC+09:00) Asia/Tokyo');

    await user.click(timezoneInput);
    await user.keyboard('Berlin');
    // By test id, not label: the label carries the current offset, which changes twice a year.
    await user.click(await screen.findByTestId(`${COMPONENT_ID}.modal.timezone.option.Europe/Berlin`));
    await save(user);

    await waitFor(() => expect(savedSchedule()['timezone']).toBe('Europe/Berlin'));
    // The wall-clock time is unchanged -- changing the zone moves the instant, not the time of day.
    expect(savedSchedule()['fireTime']).toBe('17:45');
  });

  test('offers a stored start time that is not one of the quarter hours, and keeps it', async () => {
    // The dispatcher polls every 15 minutes, so 09:08 fires late but does fire; rounding it here
    // would change when a schedule runs because someone opened the dialog.
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
    // "Daily at 16:00" does not say whether that means today or tomorrow.
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
    // The server has not recomputed it, so the stored value names a time nothing will happen at.
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
    // Tags are written through the generic endpoint, so the save asks the server for the next run
    // itself; otherwise there is nothing to show until the dispatcher's next pass 15 minutes on.
    syncedNextRunMs = Date.UTC(2026, 10, 3, 15, 0);
    experimentTags = [scheduleTag('daily', false, 'Europe/Berlin', '09:00', 0)];
    const user = setup();
    render();

    await openModal(user, /scheduled: daily/i);
    await chooseStartTime(user, '16', '00');
    await save(user);

    await waitFor(() => expect(savedSchedule()['fireTime']).toBe('16:00'));
    // A request of its own, made once the schedule is stored, so it can lag the tag write.
    await waitFor(() => expect(scheduleStateSyncs).toBe(1));
    await openModal(user, /scheduled: daily/i);
    expect(await screen.findByText(/Next run as saved:.*16:00/)).toBeInTheDocument();
  });

  test('a save still succeeds when the state cannot be brought up to date', async () => {
    // An older server has no such route, and a refusal looks the same. The schedule is already
    // written by then, and the dispatcher's next pass computes the next run anyway.
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
    // The query neither polls nor refetches on focus, so without re-reading on open the line
    // stayed missing for the rest of the session.
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
    // The server has not seen an unsaved edit, so the value must not follow the draft -- otherwise
    // the line silently answers for the controls above it.
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
    // The line describes the saved schedule, which a draft set to Off is about to stop.
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
    // Pausing a Tokyo schedule from another zone must not resume it at 09:00 local to the pauser.
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
    // the control took it. `toBeDisabled` alone would not discriminate, because the ordinary
    // trigger is disabled too while the schedule loads.
    await waitFor(async () =>
      expect(await screen.findByRole('button', { name: /schedule detection/i })).toHaveAttribute('tooltip-data-state'),
    );
    expect(await screen.findByRole('button', { name: /schedule detection/i })).toBeDisabled();

    // The wording is not asserted: the disabled button is its own tooltip trigger, and jsdom
    // delivers no pointer or focus event to a disabled element, so no unit test can open it.
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
    // Re-reading the provider's first secret on every save would re-point the schedule at
    // whichever key is listed first -- on a multi-user server, somebody else's.
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
    // An empty read says nothing about the pin; dropping it would send the next occurrence to the
    // server's own credential.
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
    // The server drops a clause carrying one, and a tab shows nothing in the field -- hence naming
    // the code point rather than echoing the value.
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
    // Named explicitly, so the dispatcher runs on this key and not whichever a later read finds.
    expect(savedSchedule()['model']).toEqual({
      provider: 'openai',
      model: 'gpt-5.6-sol',
      secretId: 'sec-new',
    });
  });

  test('names the existing secret when AI Gateway already holds one for the provider', async () => {
    // No key step, since one exists -- but the schedule must still record which one, or the
    // dispatcher is back to guessing among the provider's secrets at fire time.
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
    // Each click mints a fresh secret name, so a second lands a duplicate key. `loading` hides the
    // label but still accepts a click, so the guard has to be `disabled`. The create response is
    // held open because awaiting the first click lets the in-flight state propagate and hides it.
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

    // By component id, because the in-flight button hides its label and so its name no longer
    // matches. This is the node a second click would land on.
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
    // The secret is created before the schedule is written, and that write can fail. Minting a
    // second key on retry would leave an unused, randomly named credential per attempt.
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
    // A schedule savable before this step existed must not become unsavable because the check
    // cannot answer -- an older server, or one whose gateway is unreachable.
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
    // The cadence is what makes the dispatcher pick the schedule up, so it must land last --
    // written first, a poll in between could run against clauses not yet stored.
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

    // Against a failed read the modal would show "Off" with no filters, and saving that would
    // replace whatever is really stored.
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
      // The same trigger opens this dialog there, so the link would lead to the current page.
      experimentTags = [scheduleTag('weekly')];
      const user = setup();
      renderAt('/experiments/1/issues/runs');
      await openModal(user, /scheduled: weekly/i);

      expect(await screen.findByRole('button', { name: 'Save' })).toBeInTheDocument();
      expect(screen.queryByRole('link', { name: 'View run history' })).not.toBeInTheDocument();
    });

    test('the link is hidden on the API key step, which is mid-save', async () => {
      // The step is reached by saving, so a link here offers to walk away from an unstored
      // schedule.
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
