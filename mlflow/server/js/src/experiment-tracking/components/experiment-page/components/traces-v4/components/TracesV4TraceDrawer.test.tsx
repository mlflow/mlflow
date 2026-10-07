import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { screen, within } from '@testing-library/react';
import { rest } from 'msw';
import { setupServer } from '@mlflow/mlflow/src/common/utils/setup-msw';
import { setupTestRouter } from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { TracesV4TraceDrawer } from './TracesV4TraceDrawer';
import { renderTracesV4Page } from '../test-utils/renderTracesV4Page';
import { makeTaggedTrace, makeTrace } from '../test-utils/mockTraces';

const EXPERIMENT_ID = 'exp-1';
const PATH = '/ml/experiments/:experimentId/traces';
const URL = '/ml/experiments/exp-1/traces';

// OSS's `getTrace` (via `useGetTrace`) reads the `mlflow.trace.spansLocation` tag to decide which
// endpoint to use. When it's set to 'TRACKING_STORE', it uses the V3 endpoint; otherwise it falls
// through to the artifact route (which the test doesn't mock → drawer times out).
const TRACE_INFO_V3_ENDPOINT = '/ajax-api/3.0/mlflow/traces/:traceId';

interface ServerState {
  /** Server-side tag state for the open trace; the V3 info fetch reads it. */
  trace: ModelTraceInfoV3;
}

let state: ServerState;

const server = setupServer(
  // OSS's `getTrace` uses V3 endpoint when the trace info carries the `mlflow.trace.spansLocation`
  // tag set to `TRACKING_STORE`. This endpoint returns { trace: { trace_info, spans } }.
  rest.get(TRACE_INFO_V3_ENDPOINT, (_req, res, ctx) =>
    res(ctx.json({ trace: { trace_info: state.trace, spans: [] } })),
  ),
);

const { history } = setupTestRouter();

// On-page trace the drawer navigates within (also the source of the row's `ModelTraceInfoV3`). The
// `makeTaggedTrace` helper adds the `mlflow.trace.spansLocation: TRACKING_STORE` tag so the drawer's
// getTrace takes the V3 path.
const ON_PAGE_TRACE = makeTaggedTrace('tr-000', { env: 'prod' });
const ON_PAGE_LONG_ID = 'trace:/cat.sch/tr-000';

beforeAll(() => {
  process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
});

beforeEach(() => {
  // Fresh trace per test (makeTaggedTrace returns a new object) so tests don't leak into each other.
  state = { trace: makeTaggedTrace('tr-000', { env: 'prod' }) };
});

afterEach(() => {
  jest.clearAllMocks();
});

afterAll(() => {
  delete process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'];
});

const noop = () => {};

// The drawer body mounts the heavy shared `ModelTraceExplorer`; the first test to render it pays a
// cold module-load cost (the whole tree sits behind a page-loading Suspense fallback) that can exceed
// the default 1s `findBy` timeout. Give the dialog lookup extra headroom so results don't depend on
// which test runs first.
const findDrawer = () => screen.findByRole('dialog', {}, { timeout: 10000 });

const renderDrawer = ({ traceId }: { traceId?: string }) =>
  renderTracesV4Page({
    initialUrl: URL,
    routes: [
      {
        path: PATH,
        element: (
          <TracesV4TraceDrawer
            traceId={traceId}
            onClose={noop}
            experimentId={EXPERIMENT_ID}
            traces={[ON_PAGE_TRACE]}
            onSelectTrace={noop}
          />
        ),
      },
    ],
    history,
    experimentId: EXPERIMENT_ID,
  });

describe('TracesV4TraceDrawer', () => {
  test('opens the redesigned explorer drawer with the trace body for a selected trace', async () => {
    renderDrawer({ traceId: ON_PAGE_LONG_ID });

    const drawer = await findDrawer();
    // The redesigned (v2) header titles the drawer "Trace" and shows the id as a copyable tag (the
    // `tr-` prefix stripped, truncated to 8 chars) — there is no inline tag-edit affordance.
    expect(await within(drawer).findByRole('button', { name: '000' })).toBeInTheDocument();
    expect(within(drawer).queryByRole('button', { name: 'Edit' })).not.toBeInTheDocument();
    expect(within(drawer).queryByRole('button', { name: 'Add tags' })).not.toBeInTheDocument();
  });

  test('does not render the drawer at all when no trace is selected', () => {
    renderDrawer({ traceId: undefined });
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Edit' })).not.toBeInTheDocument();
  });

  test('opens a deep-linked trace whose row is not on the current page', async () => {
    // A deep link whose row isn't on the page carries no `ModelTraceInfoV3` from the row; the drawer
    // still resolves and renders its body via the fetched trace info.
    const { trace_location, ...infoWithoutLocation } = makeTrace('not-on-page');
    server.use(
      rest.get(TRACE_INFO_V3_ENDPOINT, (_req, res, ctx) =>
        res(ctx.json({ trace: { trace_info: infoWithoutLocation, spans: [] } })),
      ),
    );
    renderDrawer({ traceId: 'trace:/cat.sch/not-on-page' });

    const drawer = await screen.findByRole('dialog', {}, { timeout: 10000 });
    // The redesigned header never surfaces inline tag editing regardless of how the trace resolved.
    expect(within(drawer).queryByRole('button', { name: 'Edit' })).not.toBeInTheDocument();
    expect(within(drawer).queryByRole('button', { name: 'Add tags' })).not.toBeInTheDocument();
    // The full ModelTraceExplorer render is slow under parallel jsdom load; per-test timeout avoids
    // the lint-forbidden global jest.setTimeout.
  }, 15000);
});
