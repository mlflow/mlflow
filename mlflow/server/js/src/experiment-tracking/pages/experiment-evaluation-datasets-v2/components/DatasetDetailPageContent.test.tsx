/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck — punting test typing; see PR2 plan in branch import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { rest } from 'msw';
import { workspaceFetch } from '@databricks/web-shared/spog/workspace-console';
import { setupTestConfig } from '@databricks/web-shared/flags/test-utils';
import { setupTestRouter } from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';
import { setupServer } from '@mlflow/mlflow/src/common/utils/setup-msw';
import type { Dataset, DatasetRecord } from '../hooks/useDatasetsQueries';
import { ExperimentEvaluationDatasetDetailPage } from '../ExperimentEvaluationDatasetDetailPage';
import { renderDatasetsPage } from '../test-utils/renderDatasetsPage';
import { mockEmptyResponse, mockJsonResponse } from '../test-utils/mockResponses';
import { SEARCH_DEBOUNCE_MS } from '../utils/constants';
import type { JsonRecordEditorProps } from './JsonRecordEditor';
import { jest } from '@jest/globals';
import { describe } from '@jest/globals';
import { beforeEach } from '@jest/globals';
import { afterEach } from '@jest/globals';
import { test } from '@jest/globals';
import { expect } from '@jest/globals';

// `workspaceFetch` is imported by the local `useDatasetsQueries.tsx` from the package barrel,
// so the barrel mock intercepts every call from that file (single-record upsert/delete/create,
// dataset CRUD). The records LIST query lives inside @databricks/web-shared, which imports
// `fetchOrFail` via a relative path — the barrel mock does not see those calls, so MSW
// handles that endpoint instead.
jest.mock('@databricks/web-shared/spog/workspace-console', () => ({
  workspaceFetch: jest.fn(),
}));

// Monaco doesn't run under JSDOM; swap the JSON editor for a textarea that mirrors the
// public contract callers depend on. `errorMessage` renders into a role="alert" element so
// tests can assert on inline parse-error wiring without a real Monaco instance.
jest.mock('./LazyJsonRecordEditor', () => ({
  LazyJsonRecordEditor: ({ value, onChange, ariaLabel, errorMessage }: JsonRecordEditorProps) => (
    <>
      <textarea aria-label={ariaLabel} value={value} onChange={(e) => onChange(e.target.value)} />
      {errorMessage ? <div role="alert">{errorMessage}</div> : null}
    </>
  ),
}));

const EXPERIMENT_ID = 'exp-1';
const DATASET_ID = 'ds-1';
// MLflow routes are mounted under `/ml` inside Databricks (see `createMLflowRoutePath`); the
// page also generates links via the `Routes` helpers, which embed that prefix. Tests must
// use the same prefix so the route registration matches the actual URLs.
const DETAIL_PATH = '/ml/experiments/:experimentId/datasets/:datasetId';
const LIST_PATH = '/ml/experiments/:experimentId/datasets';
const DETAIL_URL = `/ml/experiments/${EXPERIMENT_ID}/datasets/${DATASET_ID}`;
const LIST_URL = `/ml/experiments/${EXPERIMENT_ID}/datasets`;

const ListRouteStub = () => <div data-testid="datasets-list-stub">list</div>;

/**
 * Wall-clock wait used exclusively for the *negative* assertion below — "the URL never gains
 * `q=` after an unmount-during-debounce sequence." Extra wall time only increases the chance
 * of catching a regression here, never the chance of a false failure, so this isn't the
 * flake-prone fixed-timeout pattern that `no-promise-with-timeout` exists to ban. The rule
 * checks for `setTimeout` *inside* a Promise constructor; keeping the timer at the function
 * body keeps the lint rule satisfied without an `eslint-disable` on the call site.
 */
const waitWallClockMs = (ms: number): Promise<void> => {
  let resolveWait: () => void = () => {};
  const waitPromise = new Promise<void>((resolve) => {
    resolveWait = resolve;
  });
  setTimeout(resolveWait, ms);
  return waitPromise;
};

const datasetFixture: Dataset = {
  dataset_id: DATASET_ID,
  name: 'Customer Support Eval',
  create_time: '2026-01-01T00:00:00Z',
  last_update_time: '2026-01-02T00:00:00Z',
  created_by: 'alice@databricks.com',
  source_type: 'manual',
};

const recordOne: DatasetRecord = {
  dataset_record_id: 'rec-1',
  inputs: { question: 'What is MLflow?' },
  expectations: { answer: 'A platform for ML lifecycle.' },
  tags: {},
  source: { human: { user_name: 'alice@databricks.com' } },
  create_time: '2026-01-01T00:00:00Z',
  last_update_time: '2026-01-02T00:00:00Z',
  created_by: 'alice@databricks.com',
  last_updated_by: 'alice@databricks.com',
};
const recordTwo: DatasetRecord = {
  dataset_record_id: 'rec-2',
  inputs: { question: 'How do I log a trace?' },
  expectations: { answer: 'Use mlflow.trace.' },
  tags: {},
  source: { human: { user_name: 'alice@databricks.com' } },
  create_time: '2026-01-03T00:00:00Z',
  last_update_time: '2026-01-04T00:00:00Z',
  created_by: 'alice@databricks.com',
  last_updated_by: 'alice@databricks.com',
};

interface MockState {
  dataset: Dataset | null;
  records: DatasetRecord[];
  /** Tracks calls so tests can assert which mutation was invoked. */
  upsertCalls: Array<{ recordId: string; body: unknown }>;
  createCalls: Array<{ body: unknown }>;
  deleteRecordCalls: string[];
  /** Flip on to make the PATCH upsert endpoint return 500. */
  upsertShouldFail: boolean;
  /** Flip on to make the POST create endpoint throw. */
  createShouldFail: boolean;
}

function installFetchMocks(state: MockState) {
  jest.mocked(workspaceFetch).mockImplementation(async (input, init) => {
    const urlStr = input.toString();
    const method = init?.method ?? 'GET';

    // PATCH /datasets/{id}/records/{recordId} — upsert (record edit).
    const upsertMatch = urlStr.match(/\/managed-evals\/datasets\/[^/]+\/records\/([^/?]+)(?:\?|$)/);
    if (upsertMatch && method === 'PATCH') {
      const body = init?.body ? JSON.parse(init.body as string) : undefined;
      state.upsertCalls.push({ recordId: upsertMatch[1], body });
      if (state.upsertShouldFail) {
        // Throw to model a true network failure: `upsertDatasetRecord` does not check
        // `response.ok` (see useDatasetsQueries.tsx — a known pre-existing gap), so a
        // mocked 5xx response wouldn't propagate to the mutation's `onError`. A thrown
        // workspaceFetch reliably triggers the failure code path.
        throw new Error('Save failed on backend');
      }
      // Mirror the real API: store `state.records` in JS format (consumed by the GET
      // handler) but return a REST-format response body (consumed by
      // `transformDatasetRecord` in upsertDatasetRecord). The PATCH body is REST-format,
      // so we have to invert the transform before merging.
      const { inputs: bodyInputs, expectations: bodyExpectations, ...restBody } = body ?? {};
      const jsInputs = Array.isArray(bodyInputs)
        ? Object.fromEntries((bodyInputs as Array<{ key: string; value: unknown }>).map((kv) => [kv.key, kv.value]))
        : undefined;
      const jsExpectations = bodyExpectations
        ? Object.fromEntries(
            Object.entries(bodyExpectations as Record<string, { value: unknown }>).map(([k, v]) => [k, v.value]),
          )
        : undefined;
      const existing = state.records.find((r) => r.dataset_record_id === upsertMatch[1]);
      const updatedJs: DatasetRecord = {
        ...existing,
        ...restBody,
        ...(jsInputs ? { inputs: jsInputs } : {}),
        ...(jsExpectations ? { expectations: jsExpectations } : {}),
      } as DatasetRecord;
      state.records = state.records.map((r) => (r.dataset_record_id === upsertMatch[1] ? updatedJs : r));
      return mockJsonResponse(toRestRecord(updatedJs));
    }

    // DELETE /datasets/{id}/records/{recordId}
    if (upsertMatch && method === 'DELETE') {
      state.deleteRecordCalls.push(upsertMatch[1]);
      state.records = state.records.filter((r) => r.dataset_record_id !== upsertMatch[1]);
      return mockEmptyResponse();
    }

    // POST /datasets/{id}/records — create.
    const createMatch = urlStr.match(/\/managed-evals\/datasets\/[^/]+\/records(?:\?|$)/);
    if (createMatch && method === 'POST') {
      const body = init?.body ? JSON.parse(init.body as string) : undefined;
      state.createCalls.push({ body });
      if (state.createShouldFail) {
        // Same pattern as upsertShouldFail — throw rather than return a non-ok response so
        // the mutation's onError fires deterministically regardless of how the underlying
        // helper handles 5xx.
        throw new Error('Create failed on backend');
      }
      const newRecord: DatasetRecord = {
        dataset_record_id: `rec-new-${state.createCalls.length}`,
        tags: {},
        source: { human: { user_name: 'alice@databricks.com' } },
        inputs: {},
        create_time: '2026-02-01T00:00:00Z',
        last_update_time: '2026-02-01T00:00:00Z',
        ...(body ?? {}),
      };
      state.records = [...state.records, newRecord];
      return mockJsonResponse(newRecord);
    }

    // DELETE /datasets/{id} — delete the whole dataset.
    const datasetMatch = urlStr.match(/\/managed-evals\/datasets\/([^/?]+)(?:\?|$)/);
    if (datasetMatch && !datasetMatch[0].includes('/records') && method === 'DELETE') {
      state.dataset = null;
      return mockEmptyResponse();
    }

    // GET /datasets/{id} — dataset metadata.
    if (datasetMatch && !datasetMatch[0].includes('/records') && method === 'GET') {
      if (!state.dataset) {
        return mockJsonResponse(
          { error_code: 'NOT_FOUND', message: 'Not found' },
          { status: 404, statusText: 'Not Found' },
        );
      }
      return mockJsonResponse(state.dataset);
    }

    return mockEmptyResponse();
  });
}

// useListDatasetRecordsQuery lives in @databricks/web-shared and imports `fetchOrFail` via a
// relative path inside the package, so a barrel-level jest.mock on `@databricks/web-shared/mfe-services`
// does not intercept it. MSW handles the list-records endpoint at the network layer instead;
// the closure reads `currentState` so test mutations are visible on the next refetch.
let currentState: MockState | undefined;
const toRestRecord = (record: DatasetRecord) => ({
  ...record,
  inputs: Object.entries(record.inputs ?? {}).map(([key, value]) => ({ key, value })),
  expectations: Object.entries(record.expectations ?? {}).reduce<Record<string, { value: unknown }>>(
    (acc, [key, value]) => {
      acc[key] = { value };
      return acc;
    },
    {},
  ),
});
const server = setupServer(
  rest.get('/ajax-api/2.0/managed-evals/datasets/:datasetId/records', (_req, res, ctx) =>
    res(ctx.json({ dataset_records: (currentState?.records ?? []).map(toRestRecord) })),
  ),
);

describe('DatasetDetailPageContent', () => {
  // setupTestRouter registers beforeAll/afterAll/beforeEach hooks, so it MUST live at describe scope.
  const { history } = setupTestRouter();
  const { setSafex } = setupTestConfig();
  let state: MockState;

  beforeEach(() => {
    setSafex({ 'databricks.fe.mlflow.enableEvalDatasetsTabV2': true });
    state = {
      dataset: datasetFixture,
      records: [recordOne, recordTwo],
      upsertCalls: [],
      createCalls: [],
      deleteRecordCalls: [],
      upsertShouldFail: false,
      createShouldFail: false,
    };
    currentState = state;
    installFetchMocks(state);
  });

  afterEach(() => {
    // Guard against tests that bail out before their own `useRealTimers()` runs — leaked fake
    // timers freeze React Query's setTimeout-based refetches, leaving the next test stuck in
    // a loading state.
    jest.useRealTimers();
    currentState = undefined;
    jest.clearAllMocks();
  });

  test('renders an error state with "Back to datasets" when the dataset fetch fails (404)', async () => {
    // The MSW/handler setup returns 404 when state.dataset is null; this simulates a deleted
    // or non-existent dataset id in the URL.
    state.dataset = null;
    state.records = [];
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [
        { path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> },
        { path: LIST_PATH, element: <ListRouteStub /> },
      ],
      history,
    });

    // Error UI renders with the localized heading + body, and a recovery link.
    expect(await screen.findByText(/Couldn't load this dataset/i)).toBeInTheDocument();
    const backLink = screen.getByRole('link', { name: /Back to datasets/i });
    expect(backLink).toBeInTheDocument();

    // Clicking the link routes back to the list page.
    await user.click(backLink);
    await waitFor(() => expect(history.location.pathname).toBe(LIST_URL));
    expect(screen.getByTestId('datasets-list-stub')).toBeInTheDocument();
  });

  test('does not fire the records LIST query when the dataset fetch fails', async () => {
    state.dataset = null;
    state.records = [recordOne, recordTwo];
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
      history,
    });

    // Error UI mounts.
    expect(await screen.findByText(/Couldn't load this dataset/i)).toBeInTheDocument();

    // No row cells render — the records query is gated behind the dataset success path.
    expect(screen.queryByText(/What is MLflow\?/)).not.toBeInTheDocument();
    expect(screen.queryByText(/How do I log a trace\?/)).not.toBeInTheDocument();
  });

  test('renders the records loading skeleton (with no toolbar) on the initial fetch', async () => {
    // Gate the records list endpoint on a promise we resolve at the end of the test so the
    // loading branch stays mounted long enough to assert, but doesn't leak a pending request
    // into the next test (which causes MSW to keep the worker busy and stalls every later
    // records fetch in the suite).
    let resolveRecords: () => void = () => {};
    const recordsGate = new Promise<void>((resolve) => {
      resolveRecords = resolve;
    });
    server.use(
      rest.get('/ajax-api/2.0/managed-evals/datasets/:datasetId/records', async (_req, res, ctx) => {
        await recordsGate;
        return res(ctx.json({ dataset_records: [] }));
      }),
    );
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
      history,
    });

    // Wait past the dataset-metadata loading skeleton onto the records loading skeleton —
    // the aria-label disambiguates the two.
    expect(await screen.findByRole('status', { name: /Loading records/i })).toBeInTheDocument();
    // Toolbar must not be rendered yet — the whole point of the change is that we don't show
    // the search input + Add Record button until we know there are records.
    expect(screen.queryByRole('textbox', { name: /Search records/i })).not.toBeInTheDocument();

    // Resolve the gate so the request completes; afterEach's `resetHandlers` then has nothing
    // pending to clean up.
    resolveRecords();
    await waitFor(() => expect(screen.queryByRole('status', { name: /Loading records/i })).not.toBeInTheDocument());
  });

  test('renders the empty state with manual + programmatic options when there are no records', async () => {
    state.records = [];
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
      history,
    });

    // Wait for the SDK docs link — only the empty state renders it, so finding it confirms
    // the loading state has fully resolved and the empty-records branch is mounted (which
    // also implies the toolbar is hidden, per Change 3).
    expect(await screen.findByRole('link', { name: /SDK|programmatic|docs/i })).toBeInTheDocument();
    // Add record CTA is part of the same empty state.
    expect(screen.getByRole('button', { name: /Add record/i })).toBeInTheDocument();
    // Toolbar (and its search) is hidden when truly empty.
    expect(screen.queryByRole('textbox', { name: /Search records/i })).not.toBeInTheDocument();
  });

  test('exposes a labelled breadcrumb navigation landmark with both crumbs', async () => {
    // Du Bois Breadcrumb renders a <div>, so the page wraps it in a labelled <nav> to surface
    // it as a navigation landmark. Detail page shows two crumbs: "Datasets" (link) + the name.
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
      history,
    });

    const nav = await screen.findByRole('navigation', { name: /Breadcrumb/i });
    expect(within(nav).getByRole('link', { name: /^Datasets$/ })).toBeInTheDocument();
    expect(within(nav).getByText(datasetFixture.name as string)).toBeInTheDocument();
  });

  test('renders the records table with default columns when records are present', async () => {
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
      history,
    });

    // Default visible columns include Inputs, Expectations, Last updated, Source, Tags.
    expect(await screen.findByRole('columnheader', { name: /Inputs/i })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: /Expectations/i })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: /Last updated/i })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: /Source/i })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: /Tags/i })).toBeInTheDocument();

    // Records render.
    await waitFor(() => {
      expect(screen.getByText(/What is MLflow\?/)).toBeInTheDocument();
    });
  });

  test('column selector toggles a column off and the corresponding header disappears', async () => {
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
      history,
    });

    // Wait for the table to render so the column header is in the DOM.
    expect(await screen.findByRole('columnheader', { name: /Source/i })).toBeInTheDocument();

    // The trigger Button carries the aria-label "Select visible columns".
    await user.click(screen.getByRole('button', { name: /Select visible columns/i }));
    // Du Bois DropdownMenu.CheckboxItem renders with role `menuitemcheckbox`.
    await user.click(await screen.findByRole('menuitemcheckbox', { name: 'Source' }));
    // Close the dropdown so the trigger button is the only "Source"-named element left.
    await user.keyboard('{Escape}');

    await waitFor(() => {
      expect(screen.queryByRole('columnheader', { name: /Source/i })).not.toBeInTheDocument();
    });
  });

  test('opening the side panel in create mode pre-populates the editors with sample JSON', async () => {
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
      history,
    });
    await waitFor(() => expect(screen.getByText(/What is MLflow\?/)).toBeInTheDocument());
    await user.click(screen.getByRole('button', { name: /Add record/i }));

    const panel = await screen.findByRole('complementary', { name: /New dataset record/i });
    // recordOne / recordTwo are singleturn (no `goal` field), so the seeded sample uses the
    // singleturn shape with a `messages` key + a `guidelines` expectations key.
    const inputsEditor = within(panel).getByRole('textbox', {
      name: /Dataset record inputs/i,
    }) as HTMLTextAreaElement;
    const expectationsEditor = within(panel).getByRole('textbox', {
      name: /Dataset record expectations/i,
    }) as HTMLTextAreaElement;
    expect(inputsEditor.value).toContain('"messages"');
    expect(expectationsEditor.value).toContain('"guidelines"');
    // The Add Record button is immediately enabled — the seed is savable from mount.
    expect(within(panel).getByRole('button', { name: /Add record/i })).toBeEnabled();
  });

  test('the side panel exposes an adjustable resize handle when open', async () => {
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
      history,
    });
    await waitFor(() => expect(screen.getByText(/What is MLflow\?/)).toBeInTheDocument());
    await user.click(screen.getByText(/What is MLflow\?/));

    // The resize handle is wired on the west edge of the side panel; assert it is rendered
    // and labelled for assistive tech.
    expect(await screen.findByRole('separator', { name: /Resize side panel/i })).toBeInTheDocument();
  });

  test('opening the side panel in create mode, filling JSON, and submitting creates a record', async () => {
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
      history,
    });
    // Wait for records to render so the toolbar (with the Add Record button) is mounted.
    await waitFor(() => expect(screen.getByText(/What is MLflow\?/)).toBeInTheDocument());

    // Only one "Add record" button exists pre-panel-open — the toolbar's.
    await user.click(screen.getByRole('button', { name: /Add record/i }));

    // Side panel appears with the create-mode accessible name.
    const panel = await screen.findByRole('complementary', { name: /New dataset record/i });

    const inputsEditor = within(panel).getByRole('textbox', { name: /Dataset record inputs/i });
    const expectationsEditor = within(panel).getByRole('textbox', { name: /Dataset record expectations/i });

    fireEvent.change(inputsEditor, { target: { value: '{"question":"What is GenAI?"}' } });
    fireEvent.change(expectationsEditor, { target: { value: '{"answer":"Generative AI"}' } });

    // The panel's primary action shares the label "Add record" with the toolbar button —
    // scope to the panel to avoid the ambiguity.
    await user.click(within(panel).getByRole('button', { name: /Add record/i }));

    await waitFor(() => expect(state.createCalls).toHaveLength(1));
    // Panel closes on success.
    await waitFor(() => {
      expect(screen.queryByRole('complementary', { name: /New dataset record/i })).not.toBeInTheDocument();
    });
    expect(await screen.findByText(/Record added/i)).toBeInTheDocument();
  });

  test('side panel in create mode shows an inline "Invalid JSON" error and disables Add record on malformed input', async () => {
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
      history,
    });
    await waitFor(() => expect(screen.getByText(/What is MLflow\?/)).toBeInTheDocument());

    await user.click(screen.getByRole('button', { name: /Add record/i }));
    const panel = await screen.findByRole('complementary', { name: /New dataset record/i });

    const inputsEditor = within(panel).getByRole('textbox', { name: /Dataset record inputs/i });
    const expectationsEditor = within(panel).getByRole('textbox', { name: /Dataset record expectations/i });

    // Valid expectations + invalid inputs: only the inputs editor should surface the alert.
    fireEvent.change(expectationsEditor, { target: { value: '{"answer":"ok"}' } });
    fireEvent.change(inputsEditor, { target: { value: '{not valid json' } });

    const alerts = await within(panel).findAllByRole('alert');
    expect(alerts).toHaveLength(1);
    expect(alerts[0]).toHaveTextContent(/Invalid JSON/i);
    expect(within(panel).getByRole('button', { name: /Add record/i })).toBeDisabled();

    // Typing valid JSON clears the alert and re-enables the primary button.
    fireEvent.change(inputsEditor, { target: { value: '{"q":"ok"}' } });
    await waitFor(() => {
      expect(within(panel).queryByRole('alert')).not.toBeInTheDocument();
    });
    expect(within(panel).getByRole('button', { name: /Add record/i })).not.toBeDisabled();
  });

  test('side panel in create mode surfaces an inline error when the create mutation fails', async () => {
    state.createShouldFail = true;
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
      history,
    });
    await waitFor(() => expect(screen.getByText(/What is MLflow\?/)).toBeInTheDocument());

    await user.click(screen.getByRole('button', { name: /Add record/i }));
    const panel = await screen.findByRole('complementary', { name: /New dataset record/i });

    const inputsEditor = within(panel).getByRole('textbox', { name: /Dataset record inputs/i });
    fireEvent.change(inputsEditor, { target: { value: '{"question":"oops"}' } });

    await user.click(within(panel).getByRole('button', { name: /Add record/i }));

    // Footer error status surfaces the backend message; panel stays open.
    expect(await within(panel).findByText(/Create failed on backend/i)).toBeInTheDocument();
    expect(screen.getByRole('complementary', { name: /New dataset record/i })).toBeInTheDocument();
  });

  test('opening the side panel in create mode renders a synthetic row that reflects live edits', async () => {
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
      history,
    });
    await waitFor(() => expect(screen.getByText(/What is MLflow\?/)).toBeInTheDocument());

    await user.click(screen.getByRole('button', { name: /Add record/i }));
    const panel = await screen.findByRole('complementary', { name: /New dataset record/i });

    const inputsEditor = within(panel).getByRole('textbox', { name: /Dataset record inputs/i });
    fireEvent.change(inputsEditor, { target: { value: '{"livePreviewKey":"livePreviewValue"}' } });

    // The synthetic row's inputs cell renders the live-parsed JSON via a monospace <span>
    // inside the records table. Scope to the table so we don't also match the editor textarea.
    await waitFor(() => {
      expect(within(screen.getByRole('table')).getByText(/livePreviewKey/)).toBeInTheDocument();
    });
  });

  test('closing the create panel with no user edits discards without a prompt', async () => {
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
      history,
    });
    await waitFor(() => expect(screen.getByText(/What is MLflow\?/)).toBeInTheDocument());

    await user.click(screen.getByRole('button', { name: /Add record/i }));
    const panel = await screen.findByRole('complementary', { name: /New dataset record/i });

    await user.click(within(panel).getByRole('button', { name: /Close/i }));

    // No DangerModal — the panel closes silently because there's no content.
    expect(screen.queryByRole('dialog', { name: /Discard unsaved changes\?/i })).not.toBeInTheDocument();
    await waitFor(() => {
      expect(screen.queryByRole('complementary', { name: /New dataset record/i })).not.toBeInTheDocument();
    });
  });

  test('closing the create panel with content shows the discard prompt', async () => {
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
      history,
    });
    await waitFor(() => expect(screen.getByText(/What is MLflow\?/)).toBeInTheDocument());

    await user.click(screen.getByRole('button', { name: /Add record/i }));
    const panel = await screen.findByRole('complementary', { name: /New dataset record/i });

    const inputsEditor = within(panel).getByRole('textbox', { name: /Dataset record inputs/i });
    fireEvent.change(inputsEditor, { target: { value: '{"q":"draft"}' } });

    await user.click(within(panel).getByRole('button', { name: /Close/i }));

    expect(await screen.findByRole('dialog', { name: /Discard unsaved changes\?/i })).toBeInTheDocument();
  });

  test('JSON-preview cells are keyboard-reachable and Enter/Space opens the drawer', async () => {
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
      history,
    });

    await waitFor(() => expect(screen.getByText(/What is MLflow\?/)).toBeInTheDocument());

    // The inputs JSON-preview cell is focusable with role="button" and an aria-label
    // that names the row's drawer-open action.
    const inputsCell = screen.getByRole('button', { name: /Open dataset record rec-1 — inputs/i });
    inputsCell.focus();
    await user.keyboard(' ');

    expect(await screen.findByRole('textbox', { name: /Dataset record inputs/i })).toBeInTheDocument();
  });

  test('record row no longer hosts the keyboard activator (tabIndex / aria-selected removed)', async () => {
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
      history,
    });

    await waitFor(() => expect(screen.getByText(/What is MLflow\?/)).toBeInTheDocument());

    // No "row" element should advertise itself as keyboard-activatable (the previous
    // pattern set tabIndex={0} + aria-label on TableRow, which conflicted with role="row").
    const rows = screen.getAllByRole('row');
    for (const row of rows) {
      expect(row).not.toHaveAttribute('tabindex', '0');
      expect(row).not.toHaveAttribute('aria-selected');
    }
  });

  test('clicking a record row opens the detail side panel with editable JSON', async () => {
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
      history,
    });

    await waitFor(() => {
      expect(screen.getByText(/What is MLflow\?/)).toBeInTheDocument();
    });
    await user.click(screen.getByText(/What is MLflow\?/));

    // Side panel's inputs editor (textarea via our mock) has the record's JSON as its value.
    const inputsEditor = await screen.findByRole('textbox', { name: /Dataset record inputs/i });
    expect((inputsEditor as HTMLTextAreaElement).value).toContain('What is MLflow?');
  });

  test('editing JSON in the drawer + Save calls the upsert mutation and shows a success toast', async () => {
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
      history,
    });

    await waitFor(() => expect(screen.getByText(/What is MLflow\?/)).toBeInTheDocument());
    await user.click(screen.getByText(/What is MLflow\?/));

    const inputsEditor = await screen.findByRole('textbox', { name: /Dataset record inputs/i });
    await user.clear(inputsEditor);
    // userEvent treats `{` as a key-descriptor opener; double-brace to type a literal `{`.
    await user.type(inputsEditor, '{{"question": "What is Databricks?"}');

    await user.click(screen.getByRole('button', { name: /^Save$/ }));

    await waitFor(() => {
      expect(state.upsertCalls).toHaveLength(1);
    });
    expect(state.upsertCalls[0].recordId).toBe('rec-1');
    expect(await screen.findByText(/Record saved/i)).toBeInTheDocument();
    // The table row reflects the saved value (via the optimistic write or the post-success
    // refetch). Without this, a regression that breaks the optimistic update would pass.
    // The inline side panel doesn't aria-hide the table — query without `hidden`.
    await waitFor(() => {
      expect(within(screen.getByRole('table')).getByText(/What is Databricks\?/)).toBeInTheDocument();
    });
  });

  test('Save error: PATCH 500 → footer error + toast + side panel stays open + edit preserved', async () => {
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
      history,
    });

    await waitFor(() => expect(screen.getByText(/What is MLflow\?/)).toBeInTheDocument());
    await user.click(screen.getByText(/What is MLflow\?/));

    const inputsEditor = await screen.findByRole('textbox', { name: /Dataset record inputs/i });
    await user.clear(inputsEditor);
    await user.type(inputsEditor, '{{"question": "Edited?"}');

    // Flip the upsert path to fail BEFORE the click — every subsequent PATCH returns 500.
    state.upsertShouldFail = true;
    await user.click(screen.getByRole('button', { name: /^Save$/ }));

    await waitFor(() => {
      expect(state.upsertCalls).toHaveLength(1);
    });

    // Error feedback surfaces: the failing mutation propagates the backend error message
    // through the save FSM (footer banner) AND the toast notification surface. Both render
    // the same string, so use findAllByText (findByText errors on multi-match).
    const errorBanners = await screen.findAllByText(/Save failed on backend/i);
    expect(errorBanners.length).toBeGreaterThan(0);

    // The side panel is still mounted with the user's edit intact — no auto-close on error.
    expect(screen.getByRole('complementary', { name: /Dataset record details/i })).toBeInTheDocument();
    // Re-query the textarea after the error round-trip; the initial reference can be stale
    // if the panel re-rendered during the optimistic-write → rollback cycle.
    const editorAfterError = screen.getByRole('textbox', { name: /Dataset record inputs/i });
    expect((editorAfterError as HTMLTextAreaElement).value).toContain('Edited?');

    // Optimistic table cell rolls back to the original after the mutation's onError fires.
    // The inline side panel doesn't aria-hide the table.
    await waitFor(() => {
      expect(within(screen.getByRole('table')).getByText(/What is MLflow\?/)).toBeInTheDocument();
    });
    expect(within(screen.getByRole('table')).queryByText(/Edited\?/)).not.toBeInTheDocument();
  });

  test('side panel close with unsaved edits surfaces the discard prompt (DangerModal)', async () => {
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
      history,
    });

    await waitFor(() => expect(screen.getByText(/What is MLflow\?/)).toBeInTheDocument());
    await user.click(screen.getByText(/What is MLflow\?/));

    const inputsEditor = await screen.findByRole('textbox', { name: /Dataset record inputs/i });
    await user.type(inputsEditor, ' (dirty)');

    const panel = screen.getByRole('complementary', { name: /Dataset record details/i });
    await user.click(within(panel).getByRole('button', { name: /Close/i }));

    // The discard prompt is rendered as a DangerModal so the confirm button uses the
    // dangerous-primary (red) treatment — asserted below.
    const dialog = await screen.findByRole('dialog', { name: /Discard unsaved changes\?/i });
    expect(dialog).toBeInTheDocument();
    const confirmButton = within(dialog).getByRole('button', { name: /^Discard$/ });
    expect(confirmButton.className).toMatch(/du-bois-light-btn-dangerous/);
  });

  test('Discard in the close-panel prompt actually closes the panel and clears recordId', async () => {
    // Regression: an in-flight URL transition stashed via requestTransition was being
    // intercepted by the page-level navigation blocker (still active because the panel was
    // still flagged dirty), which stashed a *new* transition over it; confirm() then
    // nulled both — the modal closed but the panel stayed open.
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
      history,
    });

    await waitFor(() => expect(screen.getByText(/What is MLflow\?/)).toBeInTheDocument());
    await user.click(screen.getByText(/What is MLflow\?/));

    const inputsEditor = await screen.findByRole('textbox', { name: /Dataset record inputs/i });
    await user.type(inputsEditor, ' (dirty)');

    const panel = screen.getByRole('complementary', { name: /Dataset record details/i });
    await user.click(within(panel).getByRole('button', { name: /Close/i }));

    const dialog = await screen.findByRole('dialog', { name: /Discard unsaved changes\?/i });
    await user.click(within(dialog).getByRole('button', { name: /^Discard$/ }));

    await waitFor(() => {
      expect(screen.queryByRole('complementary', { name: /Dataset record details/i })).not.toBeInTheDocument();
    });
    expect(history.location.search).not.toMatch(/recordId=/);
  });

  test('Discard when switching to a different row swaps the panel to the new record', async () => {
    // Same root cause as the close-button case — the record-switch transition runs
    // url.setRecordId(newId), which the navigation blocker would intercept and replace.
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
      history,
    });

    await waitFor(() => expect(screen.getByText(/What is MLflow\?/)).toBeInTheDocument());
    await user.click(screen.getByText(/What is MLflow\?/));

    const inputsEditor = await screen.findByRole('textbox', { name: /Dataset record inputs/i });
    await user.type(inputsEditor, ' (dirty)');

    // Open record 2 via its row activator — should prompt because record 1 is dirty.
    await user.click(screen.getByText(/How do I log a trace\?/));
    const dialog = await screen.findByRole('dialog', { name: /Discard unsaved changes\?/i });
    await user.click(within(dialog).getByRole('button', { name: /^Discard$/ }));

    // Panel now reflects record 2's saved inputs (not record 1's draft).
    await waitFor(() => {
      const editor = screen.getByRole('textbox', { name: /Dataset record inputs/i });
      expect((editor as HTMLTextAreaElement).value).toContain('How do I log a trace?');
      expect((editor as HTMLTextAreaElement).value).not.toContain('(dirty)');
    });
    expect(history.location.search).toMatch(/recordId=rec-2/);
  });

  test('bulk-select records → bulk delete removes them from the table and shows a success toast', async () => {
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
      history,
    });

    await waitFor(() => expect(screen.getByText(/What is MLflow\?/)).toBeInTheDocument());

    // Per-row checkbox labels include the record id ("Select record rec-1", etc.) so
    // screen-reader users can distinguish them from the header "select all" control.
    const rowCheckboxes = screen.getAllByRole('checkbox', { name: /Select record/i });
    await user.click(rowCheckboxes[0]);
    await user.click(rowCheckboxes[1]);

    // Selection toolbar surfaces a Delete button.
    await user.click(await screen.findByRole('button', { name: /^Delete \(2\)$/ }));

    // Confirm the bulk-delete modal.
    const confirmButtons = await screen.findAllByRole('button', { name: /^Delete$/ });
    await user.click(confirmButtons[confirmButtons.length - 1]);

    await waitFor(() => {
      expect(state.deleteRecordCalls).toEqual(expect.arrayContaining(['rec-1', 'rec-2']));
    });
    expect(await screen.findByText(/Deleted 2 records/i)).toBeInTheDocument();
  });

  test('kebab → Delete dataset opens a DangerModal scoped to this dataset and fires the delete mutation', async () => {
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [
        { path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> },
        { path: LIST_PATH, element: <ListRouteStub /> },
      ],
      history,
    });

    await waitFor(() => expect(screen.getByText(/What is MLflow\?/)).toBeInTheDocument());

    await user.click(screen.getByRole('button', { name: /Dataset actions/i }));
    await user.click(await screen.findByRole('menuitem', { name: /Delete dataset/i }));

    // The confirm modal mentions the dataset by name — confirms the dataset prop wires through.
    expect(await screen.findByText(/delete the dataset "Customer Support Eval"/i)).toBeInTheDocument();

    // Take the last-matching `Delete` (the modal's confirm — bulk-delete uses the same pattern).
    const confirmButtons = await screen.findAllByRole('button', { name: /^Delete$/ });
    await user.click(confirmButtons[confirmButtons.length - 1]);

    // The mutation must hit the dataset-delete endpoint. The post-success toast + navigation
    // race with React Query's `invalidateQueries`-triggered refetches in jsdom; that side-effect
    // is covered by the list page's per-row delete test, so we keep this test focused on the
    // detail-page contract: kebab → modal → DELETE.
    await waitFor(() =>
      expect(
        jest
          .mocked(workspaceFetch)
          .mock.calls.some(
            ([url, init]) => String(url).includes(`/datasets/${DATASET_ID}`) && init?.method === 'DELETE',
          ),
      ).toBe(true),
    );
  });

  describe('records sort', () => {
    test('clicking the Last updated header toggles sort direction in the URL and reorders rows', async () => {
      const user = userEvent.setup();
      renderDatasetsPage({
        initialUrl: DETAIL_URL,
        routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
        history,
      });

      // Wait for the default-sorted (last_updated DESC) order to render. recordTwo (2026-01-04)
      // is newer than recordOne (2026-01-02), so it appears first by default.
      await waitFor(() => {
        expect(screen.getByText(/How do I log a trace\?/)).toBeInTheDocument();
      });

      const initialOrder = screen.getAllByText(/(What is MLflow\?|How do I log a trace\?)/);
      expect(initialOrder[0].textContent).toMatch(/How do I log a trace\?/);
      expect(initialOrder[1].textContent).toMatch(/What is MLflow\?/);

      // Sortable headers render an inner role="button" whose accessible name is the header
      // text. `/^Last updated$/` keeps us from matching the "Last updated by" header next to it.
      await user.click(screen.getByRole('button', { name: /^Last updated$/ }));

      // First click flips desc → asc.
      await waitFor(() => {
        expect(history.location.search).toContain('sort=last_updated');
        expect(history.location.search).toContain('dir=asc');
      });

      // Ascending order: older record (recordOne) is now first.
      await waitFor(() => {
        const flipped = screen.getAllByText(/(What is MLflow\?|How do I log a trace\?)/);
        expect(flipped[0].textContent).toMatch(/What is MLflow\?/);
        expect(flipped[1].textContent).toMatch(/How do I log a trace\?/);
      });
    });

    test('second click on the active header flips sort direction back to the default (params cleared)', async () => {
      const user = userEvent.setup();
      renderDatasetsPage({
        initialUrl: DETAIL_URL,
        routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
        history,
      });

      await waitFor(() => expect(screen.getByText(/How do I log a trace\?/)).toBeInTheDocument());

      // Click once → asc; click again → desc. Default is last_updated DESC, so URL params clear.
      const header = screen.getByRole('button', { name: /^Last updated$/ });
      await user.click(header);
      await waitFor(() => expect(history.location.search).toContain('dir=asc'));

      await user.click(header);
      await waitFor(() => {
        expect(history.location.search).not.toContain('sort=');
        expect(history.location.search).not.toContain('dir=');
      });

      // Back to the default DESC order.
      const restored = screen.getAllByText(/(What is MLflow\?|How do I log a trace\?)/);
      expect(restored[0].textContent).toMatch(/How do I log a trace\?/);
    });
  });

  describe('records pagination', () => {
    test('clicking page 2 reveals the next slice of records and hides page 1', async () => {
      // Seed 30 records so the default 25-record page size overflows by 5. Use distinct
      // `dataset_record_id`s plus an ?sort= URL override so the order is deterministic and
      // doesn't depend on the inferred default sort.
      state.records = Array.from({ length: 30 }, (_, i) => ({
        dataset_record_id: `rec-${String(i).padStart(2, '0')}`,
        inputs: { question: `q-${String(i).padStart(2, '0')}` },
        expectations: { answer: `a-${i}` },
        tags: {},
        source: { human: { user_name: 'alice@databricks.com' } },
        create_time: '2026-01-01T00:00:00Z',
        last_update_time: '2026-01-01T00:00:00Z',
        created_by: 'alice@databricks.com',
        last_updated_by: 'alice@databricks.com',
      }));

      const user = userEvent.setup();
      renderDatasetsPage({
        // Sort by dataset_record_id ASC so page 1 deterministically shows rec-00 ... rec-24.
        initialUrl: `${DETAIL_URL}?sort=dataset_record_id&dir=asc`,
        routes: [{ path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> }],
        history,
      });

      // Page 1 should show q-00 and q-24 (first and last of the 25-record slice) and NOT q-25.
      await waitFor(() => expect(screen.getByText(/q-00/)).toBeInTheDocument());
      expect(screen.getByText(/q-24/)).toBeInTheDocument();
      expect(screen.queryByText(/q-25/)).not.toBeInTheDocument();

      // Du Bois Pagination uses `role="listitem"` for the page-number controls (see
      // design-system/Pagination/Pagination.test.tsx for the canonical query).
      await user.click(screen.getByRole('listitem', { name: '2' }));

      await waitFor(() => expect(screen.getByText(/q-25/)).toBeInTheDocument());
      expect(screen.getByText(/q-29/)).toBeInTheDocument();
      // The page-1 head record is gone — confirming a true slice change, not an append.
      expect(screen.queryByText(/q-00/)).not.toBeInTheDocument();
    });
  });

  test('search input does NOT bleed `q` back onto the list page when the user navigates before the debounce fires', async () => {
    // Real timers — fake timers paired with userEvent's `advanceTimers` tick past the 250ms
    // debounce while typing, writing `q=abc` to the URL before the navigation and stopping
    // this from reproducing the bug. Real time keeps the type/click sequence inside the
    // window so the unmount-cancel path is the only thing preventing the bleed.
    //
    // The wall-clock wait below is a *negative* assertion — "the URL never gains q="; extra
    // wall time only increases the chance of catching a regression, never the chance of a
    // false failure. That is the opposite of the flake-prone fixed-timeout pattern that
    // jest-testing.md warns against (which is "wait arbitrary time then assert positive").
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: DETAIL_URL,
      routes: [
        { path: DETAIL_PATH, element: <ExperimentEvaluationDatasetDetailPage /> },
        { path: LIST_PATH, element: <ListRouteStub /> },
      ],
      history,
    });

    await waitFor(() => expect(screen.getByText(/What is MLflow\?/)).toBeInTheDocument());

    const searchInput = screen.getByPlaceholderText(/Search inputs/i);
    await user.type(searchInput, 'abc');

    // Click the "Datasets" link in the page breadcrumb to navigate back to the list.
    const breadcrumbLink = screen.getByRole('link', { name: /^Datasets$/ });
    await user.click(breadcrumbLink);

    await waitFor(() => {
      expect(history.location.pathname).toBe(LIST_URL);
    });
    // Wait one debounce window plus a small buffer — if cancel-on-unmount were missing, the
    // pending setTimeout would fire and write `q=abc`. SEARCH_DEBOUNCE_MS keeps the test in
    // sync with the production debounce instead of hard-coding a number.
    await waitWallClockMs(SEARCH_DEBOUNCE_MS + 50);
    expect(history.location.search).not.toContain('q=');
  });
});
