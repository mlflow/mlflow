// Test-only harness (shared by the TracesV4PageContent test files); the absolute-AJAX-URL rule is
// disabled here for the same reason the eslint TEST override disables it — these URLs are msw mock
// endpoints that never run in prod. The rule's file glob only matches *.test.*, not this helper.
/* eslint-disable @mlflow/no-absolute-ajax-urls */
import { afterAll, afterEach, beforeAll, beforeEach, jest } from '@jest/globals';
import { fireEvent, screen, within } from '@testing-library/react';
import { useEffect } from 'react';
import type userEvent from '@testing-library/user-event';
import { rest } from 'msw';
import { setupServer } from '@mlflow/mlflow/src/common/utils/setup-msw';
import { setupTestRouter } from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';
import { useLocation } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { TracesV4PageContent } from '../components/TracesV4PageContent';
import { renderTracesV4Page, seedWarehouse } from './renderTracesV4Page';
import { makeTraces } from './mockTraces';

export const EXPERIMENT_ID = 'exp-1';
const PATH = '/ml/experiments/:experimentId/traces';
export const URL = '/ml/experiments/exp-1/traces';

// OSS uses the synchronous V3 search endpoint, which returns { traces: [...], next_page_token }.
export const SEARCH_ENDPOINT = '/ajax-api/3.0/mlflow/traces/search';

export interface SearchCall {
  filter?: string;
  order_by?: string[];
  max_results?: number;
  page_token?: string;
  locations?: {
    type?: string;
    uc_schema?: { catalog_name?: string; schema_name?: string };
    mlflow_experiment?: { experiment_id?: string };
  }[];
  // Progressive-transport request fields (singular location, page_size, required client).
  location?: {
    type?: string;
    uc_schema?: { catalog_name?: string; schema_name?: string };
    uc_table_prefix?: { catalog_name?: string; schema_name?: string; table_prefix?: string };
  };
  page_size?: number;
  client?: string;
}

// Mutable server state so handlers reflect per-test setup (cursor pages, failures).
export interface ServerState {
  // Map from incoming page_token (or '' for the first page) → { traces, next_page_token }.
  pages: Record<string, { traces: ReturnType<typeof makeTraces>; next_page_token?: string }>;
  searchCalls: SearchCall[];
  searchShouldFail: boolean;
  // Server-supplied error message when a search fails (surfaced by fetchAPI as `err.message`).
  searchErrorMessage: string;
}

// Stable mutable state objects — NOT reassigned in beforeEach, but reset in place
export const state: ServerState = {
  pages: {},
  searchCalls: [],
  searchShouldFail: false,
  searchErrorMessage: '',
};

// Mutable env vars that previously lived as bare reassigned primitives
export const env = {
  lastSearch: '',
  metricsTotalCount: 42,
};

const LocationSpy = () => {
  const search = useLocation().search;
  useEffect(() => {
    env.lastSearch = search;
  }, [search]);
  return null;
};

export const renderPage = ({ initialUrl = URL }: { initialUrl?: string } = {}) => {
  env.lastSearch = '';
  return renderTracesV4Page({
    initialUrl,
    routes: [
      {
        path: PATH,
        element: (
          <>
            <LocationSpy />
            <TracesV4PageContent experimentId={EXPERIMENT_ID} />
          </>
        ),
      },
    ],
    history,
    experimentId: EXPERIMENT_ID,
  });
};

// Trace ID is hidden by default, so locate a row by its linked Input cell (a default-visible column).
export const findTraceRow = (traceId: string) => screen.findByRole('link', { name: `Open trace ${traceId} — input` });
export const queryTraceRow = (traceId: string) => screen.queryByRole('link', { name: `Open trace ${traceId} — input` });

// AntD's `onPressEnter` (wired to commit the search) is gated on the legacy `keyCode === 13`, which
// userEvent's keyboard synthesis doesn't set — fire keyDown directly to exercise that path.
export const pressEnter = (input: HTMLElement) =>
  fireEvent.keyDown(input, { key: 'Enter', code: 'Enter', keyCode: 13 });

// Open the Display popover and expand one of its submenus (Columns / Sort / Row height). The column
// controls, sort, and row-height now live behind this popover instead of standalone toolbar buttons.
// Radix submenus open on click of their SubTrigger (role="menuitem"). The Sort and Row-height triggers
// append their current value as a hint (e.g. "SortTime"), so match by the leading label via regex.
export const openDisplaySubmenu = async (user: ReturnType<typeof userEvent.setup>, submenu: RegExp) => {
  await user.click(screen.getByRole('button', { name: 'Display' }));
  await user.click(await screen.findByRole('menuitem', { name: submenu }));
};

// Click a checkbox/radio item inside an open Display submenu. Radix menu items in a SubContent don't
// respond to userEvent's realistic pointer sequence under jsdom (the pointer events get swallowed), so
// fire the click directly — same pragmatic workaround as `pressEnter` above for AntD's keyCode gate.
export const selectSubmenuItem = async (role: 'menuitemcheckbox' | 'menuitemradio', name: string) => {
  fireEvent.click(await screen.findByRole(role, { name }));
};

// Sort ascending (opposite table default) to guarantee refetch. Also tests column header sort.
export const sortByTime = async (
  user: ReturnType<typeof userEvent.setup>,
  direction: 'ascending' | 'descending' = 'ascending',
) => {
  const timeHeader = screen.getByRole('columnheader', { name: 'Time' });
  await user.click(within(timeHeader).getByRole('button', { name: 'Column options' }));
  await user.click(await screen.findByRole('menuitem', { name: `Sort ${direction}` }));
};

export const server = setupServer(
  // OSS uses the synchronous V3 search endpoint. Record the request body (assertions target this),
  // look up the page by token, and return { traces, next_page_token }.
  rest.post(SEARCH_ENDPOINT, async (req, res, ctx) => {
    const body = (await req.json()) as SearchCall;
    state.searchCalls.push(body);
    if (state.searchShouldFail) {
      return res(ctx.status(500), ctx.json({ message: state.searchErrorMessage }));
    }
    const pageToken = body.page_token ?? '';
    const page = state.pages[pageToken] ?? { traces: [], next_page_token: undefined };
    return res(ctx.json({ traces: page.traces, next_page_token: page.next_page_token }));
  }),
  // The add-to-dataset provider (GenAITracesTableProvider / ExportTracesToDatasetModal) fetches the
  // datasets list on mount; stub it so an unhandled request doesn't disrupt the toolbar render.
  rest.post('/ajax-api/3.0/mlflow/datasets/search', (_req, res, ctx) => res(ctx.json({ datasets: [] }))),
  // The footer "{n} of {total}" count reads the total from the trace-metrics endpoint. The endpoint
  // version depends on `shouldUseTracesV4API()` (3.0 when off, 4.0 when on), so mock both — the V4
  // tab uses 4.0 when the flag is enabled.
  rest.post('/ajax-api/3.0/mlflow/traces/metrics', (_req, res, ctx) =>
    res(ctx.json({ data_points: [{ metric_name: 'trace_count', values: { COUNT: env.metricsTotalCount } }] })),
  ),
  rest.post('/ajax-api/4.0/mlflow/traces/metrics', (_req, res, ctx) =>
    res(ctx.json({ data_points: [{ metric_name: 'trace_count', values: { COUNT: env.metricsTotalCount } }] })),
  ),
);

export const { history } = setupTestRouter();

beforeAll(() => {
  process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
  server.listen();
});

beforeEach(() => {
  seedWarehouse(EXPERIMENT_ID);
  // Mark the Detect Issues first-visit guidance as already seen. Its popover renders `modal` (open by
  // default when unseen), which sets `aria-hidden` on the rest of the page and hides the trace rows
  // from queries — the same returning-user state real sessions land in after the first dismissal.
  window.localStorage.setItem('mlflow.detectIssues.guidanceShown_v1', 'true');
  // Reset state in place (not reassigned const)
  state.pages = { '': { traces: makeTraces(3), next_page_token: undefined } };
  state.searchCalls = [];
  state.searchShouldFail = false;
  state.searchErrorMessage = 'search boom';
  env.metricsTotalCount = 42;
});

afterEach(() => {
  jest.useRealTimers();
  jest.clearAllMocks();
  window.localStorage.clear();
});

afterAll(() => {
  delete process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'];
  server.close();
});
