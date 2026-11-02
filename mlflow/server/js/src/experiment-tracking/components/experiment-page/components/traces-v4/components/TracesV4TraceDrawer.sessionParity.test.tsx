import { afterAll, afterEach, beforeAll, describe, expect, jest, test } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { rest } from 'msw';
import { setupServer } from '@mlflow/mlflow/src/common/utils/setup-msw';
import { setupTestRouter } from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { TracesV4TraceDrawer, type TracesV4TraceDrawerSession } from './TracesV4TraceDrawer';
import { renderTracesV4Page } from '../test-utils/renderTracesV4Page';
import { makeTrace } from '../test-utils/mockTraces';

// The session page let judges run on the whole session; the drawer's session view keeps that.
jest.mock(
  '@mlflow/mlflow/src/experiment-tracking/pages/experiment-scorers/hooks/useRunScorerInTracesViewConfiguration',
  () => ({
    useRunScorerInTracesViewConfiguration: () => ({
      renderRunJudgeModal: () => null,
      subscribeToScorerFinished: () => () => {},
      evaluateTraces: jest.fn(),
      allEvaluations: {},
    }),
  }),
);

const EXPERIMENT_ID = 'exp-1';
const PATH = '/ml/experiments/:experimentId/traces';
const URL = '/ml/experiments/exp-1/traces';
const SESSION_ID = 'session-abc';

const makeTurn = (id: string, requestTime: string): ModelTraceInfoV3 =>
  makeTrace(id, {
    trace_location: { type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: EXPERIMENT_ID } },
    request_time: requestTime,
    tags: { 'mlflow.trace.spansLocation': 'TRACKING_STORE' },
    trace_metadata: { 'mlflow.trace.session': SESSION_ID },
  });
const TURNS = [
  makeTurn('tr-turn-3', '2025-01-01T00:02:00.000Z'),
  makeTurn('tr-turn-1', '2025-01-01T00:00:00.000Z'),
  makeTurn('tr-turn-2', '2025-01-01T00:01:00.000Z'),
];
// A page of traces that isn't the session's turn order, so page navigation and turn navigation differ.
const PAGE_TRACES = [TURNS[0], makeTrace('tr-other'), TURNS[1], TURNS[2]];

const searchFilters: string[] = [];
const server = setupServer(
  rest.post('/ajax-api/3.0/mlflow/traces/search', async (req, res, ctx) => {
    searchFilters.push((await req.json()).filter);
    return res(ctx.json({ traces: TURNS }));
  }),
  rest.get('/ajax-api/3.0/mlflow/traces/get', (req, res, ctx) => {
    const traceInfo = TURNS.find((turn) => turn.trace_id === req.url.searchParams.get('trace_id'));
    return res(ctx.json({ trace: { trace_info: traceInfo, spans: [] } }));
  }),
);

const { history } = setupTestRouter();

beforeAll(() => {
  process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
  Element.prototype.scrollTo = jest.fn() as unknown as Element['scrollTo'];
  server.listen();
});

afterEach(() => {
  jest.clearAllMocks();
  searchFilters.length = 0;
});

afterAll(() => {
  delete process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'];
  server.close();
});

const makeSession = (over: Partial<TracesV4TraceDrawerSession> = {}): TracesV4TraceDrawerSession => ({
  sessionId: SESSION_ID,
  viewMode: 'session',
  onViewTrace: jest.fn(),
  onBackToSession: jest.fn(),
  ...over,
});

const renderDrawer = ({
  traceId,
  session,
  onSelectTrace = jest.fn(),
}: {
  traceId?: string;
  session?: TracesV4TraceDrawerSession;
  onSelectTrace?: (traceId: string) => void;
}) =>
  renderTracesV4Page({
    initialUrl: URL,
    routes: [
      {
        path: PATH,
        element: (
          <TracesV4TraceDrawer
            traceId={traceId}
            onClose={jest.fn()}
            experimentId={EXPERIMENT_ID}
            traces={PAGE_TRACES}
            onSelectTrace={onSelectTrace}
            {...(session ? { session } : {})}
          />
        ),
      },
    ],
    history,
    experimentId: EXPERIMENT_ID,
  });

describe('TracesV4TraceDrawer session parity with the session page', () => {
  test("a session's trace moves between the session's turns, oldest first", async () => {
    const session = makeSession({ viewMode: 'trace' });
    renderDrawer({ traceId: 'tr-turn-2', session });

    await screen.findByText('Trace from session', {}, { timeout: 10000 });
    await waitFor(() => expect(screen.getByRole('button', { name: 'Next trace' })).toBeEnabled(), { timeout: 10000 });
    expect(searchFilters.some((filter) => filter.includes(SESSION_ID))).toBe(true);

    await userEvent.click(screen.getByRole('button', { name: 'Next trace' }));
    expect(session.onViewTrace).toHaveBeenLastCalledWith('tr-turn-3');
    await userEvent.click(screen.getByRole('button', { name: 'Previous trace' }));
    expect(session.onViewTrace).toHaveBeenLastCalledWith('tr-turn-1');
  });

  test('the first and last turns disable the matching arrow', async () => {
    const session = makeSession({ viewMode: 'trace' });
    renderDrawer({ traceId: 'tr-turn-1', session });

    await waitFor(() => expect(screen.getByRole('button', { name: 'Next trace' })).toBeEnabled(), { timeout: 10000 });
    expect(screen.getByRole('button', { name: 'Previous trace' })).toBeDisabled();
  });

  test('shows the session totals and a copy button for the session id', async () => {
    renderDrawer({ traceId: 'tr-turn-1', session: makeSession() });

    expect(await screen.findByText('3 turns', {}, { timeout: 10000 })).toBeInTheDocument();
    expect(screen.getByText('Latency')).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: /copy/i }).length).toBeGreaterThan(0);
  });

  test('judges can run on the whole session from its assessments pane', async () => {
    renderDrawer({ traceId: 'tr-turn-1', session: makeSession() });

    await screen.findByText('3 turns', {}, { timeout: 10000 });
    await userEvent.click(screen.getByRole('button', { name: /^Assess trace/ }));
    await userEvent.click(await screen.findByRole('button', { name: /Add feedback/ }));
    expect(await screen.findByText('LLM judge feedback')).toBeInTheDocument();
  });

  test('without a session the arrows still move through the page and no session is loaded', async () => {
    const onSelectTrace = jest.fn();
    renderDrawer({ traceId: 'tr-turn-3', onSelectTrace });

    await waitFor(() => expect(screen.getByRole('button', { name: 'Next trace' })).toBeEnabled(), { timeout: 10000 });
    await userEvent.click(screen.getByRole('button', { name: 'Next trace' }));
    expect(onSelectTrace).toHaveBeenLastCalledWith('trace:/cat.sch/tr-other');
    expect(searchFilters.some((filter) => filter.includes(SESSION_ID))).toBe(false);
  });
});
