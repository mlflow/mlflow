import { beforeEach, describe, expect, test } from '@jest/globals';
import { rest } from 'msw';
import userEvent, { PointerEventsCheckLevel } from '@testing-library/user-event';

import { renderWithDesignSystem, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { setupServer } from '../../../common/utils/setup-msw';
import { QueryClient, QueryClientProvider } from '../../../common/utils/reactQueryHooks';
import { MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG, MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER_TAG } from '../../constants';
import { ExperimentIssuesScheduleControl } from './ExperimentIssuesScheduleControl';
import { ISSUE_DETECTION_SCHEDULE_VERSION } from './issueDetectionSchedule';

interface SavedTag {
  key: string;
  value: string;
}

let experimentTags: SavedTag[];
let written: SavedTag[];
let saveFails: boolean;
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
  rest.get('*/ajax-api/2.0/mlflow/experiments/get', (_req, res, ctx) =>
    res(ctx.json({ experiment: { experiment_id: '1', name: 'exp', tags: experimentTags } })),
  ),
  rest.post('*/ajax-api/2.0/mlflow/experiments/set-experiment-tag', async (req, res, ctx) => {
    if (saveFails) {
      return res(ctx.status(500), ctx.json({ message: 'nope' }));
    }
    const body = (await req.json()) as SavedTag;
    written.push({ key: body.key, value: body.value });
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
  rest.post('*/ajax-api/3.0/mlflow/gateway/secrets/create', async (req, res, ctx) => {
    const body = (await req.json()) as (typeof createdSecrets)[number];
    createdSecrets.push(body);
    const secret = { secret_id: 'sec-new', secret_name: body.secret_name, provider: body.provider ?? '' };
    gatewaySecrets = [...gatewaySecrets, secret];
    return res(ctx.json({ secret }));
  }),
);

const scheduleTag = (cadence: string | null, paused = false): SavedTag => ({
  key: MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG,
  value: JSON.stringify({
    version: ISSUE_DETECTION_SCHEDULE_VERSION,
    cadence,
    timezone: 'UTC',
    paused,
  }),
});

const traceFilterTag = (filters: unknown[]): SavedTag => ({
  key: MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER_TAG,
  value: JSON.stringify(filters),
});

const render = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, cacheTime: 0 } } });
  return renderWithDesignSystem(
    <QueryClientProvider client={client}>
      <ExperimentIssuesScheduleControl experimentId="1" />
    </QueryClientProvider>,
  );
};

const setup = () => userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

const UNATTENDED_RUN_APPROVAL = /I understand that scheduled runs analyze traces and spend unattended/;

const savedSchedule = (): Record<string, unknown> =>
  JSON.parse(written.find(({ key }) => key === MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG)?.value ?? '{}');

const savedTraceFilters = (): unknown[] =>
  JSON.parse(written.find(({ key }) => key === MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER_TAG)?.value ?? 'null');

const openModal = async (user: ReturnType<typeof setup>, name = /schedule detection/i) => {
  const trigger = await screen.findByRole('button', { name });
  await waitFor(() => expect(trigger).not.toBeDisabled());
  await user.click(trigger);
  return screen.findByRole('dialog');
};

const chooseCadence = async (user: ReturnType<typeof setup>, cadence: string) => {
  await user.click(await screen.findByRole('combobox', { name: /cadence/i }));
  await user.click(await screen.findByRole('option', { name: cadence }));
};

const approveUnattendedRun = async (user: ReturnType<typeof setup>) =>
  user.click(await screen.findByRole('checkbox', { name: UNATTENDED_RUN_APPROVAL }));

const save = async (user: ReturnType<typeof setup>) => user.click(await screen.findByRole('button', { name: 'Save' }));

describe('ExperimentIssuesScheduleControl', () => {
  beforeEach(() => {
    experimentTags = [];
    written = [];
    saveFails = false;
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
    await approveUnattendedRun(user);
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
    await approveUnattendedRun(user);
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
    await chooseCadence(user, 'Hourly');

    await user.click(await screen.findByRole('button', { name: 'Add filter' }));
    await user.type(await screen.findByRole('textbox', { name: 'Filter key' }), 'environment');
    await user.type(await screen.findByRole('textbox', { name: 'Filter value' }), 'production');
    await approveUnattendedRun(user);
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

    await approveUnattendedRun(user);
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
    await approveUnattendedRun(user);
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
    await chooseCadence(user, 'Hourly');

    await user.click(await screen.findByRole('button', { name: 'Add filter' }));
    await user.type(await screen.findByRole('textbox', { name: 'Filter value' }), 'production');
    await approveUnattendedRun(user);
    await save(user);

    await waitFor(() => expect(savedTraceFilters()).toEqual([]));
  });

  test('saves a schedule only after the owner approves what the unattended run does', async () => {
    const user = setup();
    render();
    await openModal(user);
    await chooseCadence(user, 'Daily');

    expect(await screen.findByRole('button', { name: 'Save' })).toBeDisabled();
    await approveUnattendedRun(user);
    expect(await screen.findByRole('button', { name: 'Save' })).not.toBeDisabled();
    await save(user);
    await waitFor(() => expect(savedSchedule()['cadence']).toBe('daily'));
  });

  test('asks for approval again each time the modal opens, even for a schedule already on', async () => {
    experimentTags = [scheduleTag('hourly')];
    const user = setup();
    render();

    await openModal(user, /scheduled: hourly/i);
    expect(await screen.findByRole('checkbox', { name: UNATTENDED_RUN_APPROVAL })).not.toBeChecked();
    expect(await screen.findByRole('button', { name: 'Save' })).toBeDisabled();
  });

  test('turns a schedule off without asking for approval', async () => {
    experimentTags = [scheduleTag('hourly')];
    const user = setup();
    render();

    await openModal(user, /scheduled: hourly/i);
    await chooseCadence(user, 'Off');
    expect(screen.queryByRole('checkbox', { name: UNATTENDED_RUN_APPROVAL })).not.toBeInTheDocument();
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
    await approveUnattendedRun(user);

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

    const user = setup();
    await waitFor(async () =>
      expect(await screen.findByRole('button', { name: /schedule detection/i })).toBeDisabled(),
    );
    // pointerEventsCheck is off so hover reaches the disabled button; the tooltip states the
    // reason. The hover is retried because the tooltip opens on a delay the first pointer event
    // can land inside, which made a single hover flaky under a loaded suite.
    await waitFor(
      async () => {
        await user.hover(await screen.findByRole('button', { name: /schedule detection/i }));
        expect(await screen.findByRole('tooltip')).toHaveTextContent(/cannot run scheduled detection/i);
      },
      { timeout: 5000 },
    );
  });

  test('saves the model the picker is left on, defaulting to the first provider', async () => {
    const user = setup();
    render();
    await openModal(user);
    expect(await screen.findByTestId('model-dropdown-trigger')).toHaveTextContent('OpenAI');
    await chooseCadence(user, 'Daily');
    await approveUnattendedRun(user);
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
    await approveUnattendedRun(user);
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

  test('asks for an API key before saving when neither the server nor AI Gateway has one', async () => {
    envCredentialsAvailable = false;
    const user = setup();
    render();
    await openModal(user);
    await chooseCadence(user, 'Daily');
    await approveUnattendedRun(user);
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
    // The key is referenced by provider, not stored in the schedule, so rotating it keeps running.
    expect(savedSchedule()['model']).toEqual({ provider: 'openai', model: 'gpt-5.6-sol' });
  });

  test('does not ask for a key when AI Gateway already holds one for the provider', async () => {
    envCredentialsAvailable = false;
    gatewaySecrets = [{ secret_id: 'sec-1', secret_name: 'openai-key', provider: 'openai' }];
    const user = setup();
    render();
    await openModal(user);
    await chooseCadence(user, 'Daily');
    await approveUnattendedRun(user);
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
    await approveUnattendedRun(user);
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
    await approveUnattendedRun(user);
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
    await approveUnattendedRun(user);
    await save(user);

    expect(await screen.findByTestId('schedule-api-key-view')).toBeInTheDocument();
    await user.click(await screen.findByRole('button', { name: 'Back' }));

    expect(await screen.findByRole('combobox', { name: /cadence/i })).toHaveTextContent('Daily');
    expect(await screen.findByRole('checkbox', { name: UNATTENDED_RUN_APPROVAL })).toBeChecked();
  });
});
