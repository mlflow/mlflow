import { afterAll, afterEach, beforeAll, describe, expect, jest, test } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { rest } from 'msw';
import { setupServer } from '@mlflow/mlflow/src/common/utils/setup-msw';
import { setupTestRouter } from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { TracesV4TraceDrawer, type TracesV4TraceDrawerSession } from './TracesV4TraceDrawer';
import { renderTracesV4Page } from '../test-utils/renderTracesV4Page';
import { makeTrace } from '../test-utils/mockTraces';

const EXPERIMENT_ID = 'exp-1';
const PATH = '/ml/experiments/:experimentId/traces';
const URL = '/ml/experiments/exp-1/traces';
const SESSION_ID = 'session-abc';

// Experiment traces whose spans live in the tracking store, so the drawer reads them from the V3 API.
const makeTurn = (id: string, requestTime: string): ModelTraceInfoV3 =>
  makeTrace(id, {
    trace_location: { type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: EXPERIMENT_ID } },
    request_time: requestTime,
    tags: { 'mlflow.trace.spansLocation': 'TRACKING_STORE' },
    trace_metadata: { 'mlflow.trace.session': SESSION_ID },
  });
const TURNS = [makeTurn('tr-turn-2', '2025-01-01T00:01:00.000Z'), makeTurn('tr-turn-1', '2025-01-01T00:00:00.000Z')];

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
  // jsdom has no element scrolling; the drawer scrolls the selected turn into view.
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

const renderDrawer = ({ traceId, session }: { traceId?: string; session: TracesV4TraceDrawerSession }) =>
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
            traces={TURNS}
            onSelectTrace={jest.fn()}
            session={session}
          />
        ),
      },
    ],
    history,
    experimentId: EXPERIMENT_ID,
  });

describe('TracesV4TraceDrawer sessions', () => {
  test('shows the session conversation, its turns in order, and opens a turn’s full trace', async () => {
    const session = makeSession();
    renderDrawer({ traceId: 'tr-turn-1', session });

    expect(await screen.findByRole('dialog', {}, { timeout: 10000 })).toBeInTheDocument();
    expect(screen.getByText('Session')).toBeInTheDocument();
    expect(await screen.findByText('2 turns', {}, { timeout: 10000 })).toBeInTheDocument();
    expect(searchFilters[0]).toContain(SESSION_ID);

    // One "View full trace" action per turn, oldest first.
    const viewTraceButtons = await screen.findAllByRole('button', { name: 'View full trace' });
    expect(viewTraceButtons).toHaveLength(2);
    await userEvent.click(viewTraceButtons[0]);
    expect(session.onViewTrace).toHaveBeenLastCalledWith('tr-turn-1');
    await userEvent.click(viewTraceButtons[1]);
    expect(session.onViewTrace).toHaveBeenLastCalledWith('tr-turn-2');
  });

  test('moves between sessions with the drawer arrows', async () => {
    const session = makeSession({ onNextSession: jest.fn() });
    renderDrawer({ traceId: 'tr-turn-1', session });

    await screen.findByRole('dialog', {}, { timeout: 10000 });
    expect(screen.getByRole('button', { name: /previous/i })).toBeDisabled();
    await userEvent.click(screen.getByRole('button', { name: /next/i }));
    expect(session.onNextSession).toHaveBeenCalled();
  });

  test('shows a trace from the session with a way back to the conversation', async () => {
    const session = makeSession({ viewMode: 'trace' });
    renderDrawer({ traceId: 'tr-turn-1', session });

    expect(await screen.findByText('Trace from session', {}, { timeout: 10000 })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('link', { name: 'Back to session' }));
    expect(session.onBackToSession).toHaveBeenCalled();
  });
});
