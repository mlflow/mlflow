import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import { useEffect } from 'react';
import userEvent, { PointerEventsCheckLevel } from '@testing-library/user-event';
import { rest } from 'msw';
import { setupServer } from '@mlflow/mlflow/src/common/utils/setup-msw';
import { setupTestRouter } from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';
import { useLocation } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { slowlyTypeEachKey } from '@databricks/web-shared/test-utils/slowlyTypeEachKey';
import { MlflowService } from '@mlflow/mlflow/src/experiment-tracking/sdk/MlflowService';
import { TracesV4PageContent } from './TracesV4PageContent';
import { renderTracesV4Page, seedWarehouse } from '../test-utils/renderTracesV4Page';
import {
  makeFeedbackAssessment,
  makeSessionTrace,
  makeTaggedTrace,
  makeTrace,
  makeTraces,
} from '../test-utils/mockTraces';
import { TRACE_COLUMN_SIZES_STORAGE_KEY_PREFIX } from '../utils/constants';
import { COLUMN_SIZES_STORAGE_VERSION } from '../hooks/useTracesV4ColumnSizing';
import { setLocalStorageItem } from '@databricks/web-shared/hooks';

// The saved-views hook (mounted via the toolbar's Views button) reads experiment tags through the
// Apollo experiment query. These page tests exercise the table/filter/pagination flows, not saved
// views, so stub it with a stable empty-tags result — leaving it unmocked fires a real Apollo query
// with no GraphQL handler, whose async failure slows the heavy filter-interaction tests past their
// timeout. The saved-views behavior itself is covered in TracesV4SavedViews.test.tsx.
jest.mock('@mlflow/mlflow/src/experiment-tracking/hooks/useExperimentQuery', () => ({
  useGetExperimentQuery: () => ({ data: { tags: [] }, refetch: () => Promise.resolve({}) }),
}));

const EXPERIMENT_ID = 'exp-1';
const STORAGE_UC_SCHEMA = 'cat.sch';
const PATH = '/ml/experiments/:experimentId/traces';
const URL = '/ml/experiments/exp-1/traces';

// OSS uses the synchronous V3 search endpoint, which returns { traces: [...], next_page_token }.
const SEARCH_ENDPOINT = '/ajax-api/3.0/mlflow/traces/search';

interface SearchCall {
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
interface ServerState {
  // Map from incoming page_token (or '' for the first page) → { traces, next_page_token }.
  pages: Record<string, { traces: ReturnType<typeof makeTraces>; next_page_token?: string }>;
  searchCalls: SearchCall[];
  searchShouldFail: boolean;
  // Server-supplied error message when a search fails (surfaced by fetchAPI as `err.message`).
  searchErrorMessage: string;
}

let state: ServerState;

let lastSearch = '';

// Total returned by the trace-metrics endpoint for the footer "{n} of {total}" count.
let metricsTotalCount = 42;

const LocationSpy = () => {
  const search = useLocation().search;
  useEffect(() => {
    lastSearch = search;
  }, [search]);
  return null;
};

const renderPage = ({ initialUrl = URL }: { initialUrl?: string } = {}) => {
  lastSearch = '';
  return renderTracesV4Page({
    initialUrl,
    routes: [
      {
        path: PATH,
        element: (
          <>
            <LocationSpy />
            <TracesV4PageContent experimentId={EXPERIMENT_ID} storageUCSchema={STORAGE_UC_SCHEMA} />
          </>
        ),
      },
    ],
    history,
    experimentId: EXPERIMENT_ID,
  });
};

// Trace ID is hidden by default, so locate a row by its Input cell activator (a default-visible
// role="button" that opens the drawer). Its accessible name is "Open trace <id> — input".
const findTraceRow = (traceId: string) => screen.findByRole('button', { name: `Open trace ${traceId} — input` });
const queryTraceRow = (traceId: string) => screen.queryByRole('button', { name: `Open trace ${traceId} — input` });

// AntD's `onPressEnter` (wired to commit the search) is gated on the legacy `keyCode === 13`, which
// userEvent's keyboard synthesis doesn't set — fire keyDown directly to exercise that path.
const pressEnter = (input: HTMLElement) => fireEvent.keyDown(input, { key: 'Enter', code: 'Enter', keyCode: 13 });

const server = setupServer(
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
  // The footer "{n} of {total}" count reads the total from the trace-metrics endpoint.
  rest.post('/ajax-api/3.0/mlflow/traces/metrics', (_req, res, ctx) =>
    res(ctx.json({ data_points: [{ metric_name: 'trace_count', values: { COUNT: metricsTotalCount } }] })),
  ),
);

const { history } = setupTestRouter();

beforeAll(() => {
  process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
});

beforeEach(() => {
  seedWarehouse(EXPERIMENT_ID);
  // Mark the Detect Issues first-visit guidance as already seen. Its popover renders `modal` (open by
  // default when unseen), which sets `aria-hidden` on the rest of the page and hides the trace rows
  // from queries — the same returning-user state real sessions land in after the first dismissal.
  window.localStorage.setItem('mlflow.detectIssues.guidanceShown_v1', 'true');
  metricsTotalCount = 42;
  state = {
    pages: { '': { traces: makeTraces(3), next_page_token: undefined } },
    searchCalls: [],
    searchShouldFail: false,
    searchErrorMessage: 'search boom',
  };
});

afterEach(() => {
  jest.useRealTimers();
  jest.clearAllMocks();
  window.localStorage.clear();
});

afterAll(() => {
  delete process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'];
});

describe('TracesV4PageContent', () => {
  test('renders a row per trace with the default columns', async () => {
    renderPage();
    expect(await findTraceRow('tr-000')).toBeInTheDocument();
    expect(screen.getByText('request for tr-000')).toBeInTheDocument();
    expect(screen.getByText('response for tr-000')).toBeInTheDocument();
    // Default columns: Time, Input, Output, Duration, State (no Session on this session-less page).
    expect(screen.getByRole('columnheader', { name: 'Time' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'State' })).toBeInTheDocument();
    // Trace ID and Tokens are hidden by default (available via the column selector).
    expect(screen.queryByRole('columnheader', { name: 'Trace ID' })).not.toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'Tokens' })).not.toBeInTheDocument();
    // Per-test timeout (matching the file's other heavy renders): the full page render is slow under
    // parallel jsdom load and would otherwise flake against the default 5s ceiling.
  }, 20000);

  test('enabling Trace ID puts it as the first data column, left of Time', async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderPage();
    await findTraceRow('tr-000');

    await user.click(screen.getByRole('button', { name: 'Select visible columns' }));
    await user.click(await screen.findByRole('menuitemcheckbox', { name: 'Trace ID' }));

    const traceId = await screen.findByRole('columnheader', { name: 'Trace ID' });
    const time = screen.getByRole('columnheader', { name: 'Time' });
    // DOCUMENT_POSITION_FOLLOWING (4) means Time comes after Trace ID in document order — i.e. Trace
    // ID is the leftmost data column (left of Time). The row-select cell renders before both.
    expect(traceId.compareDocumentPosition(time) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  }, 20000);

  test('Next sends the second page token and shows the second page; Prev does not refetch', async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    // Page 1 must be a *full* page (rows === pageSize) for Next to enable — a short page is treated as
    // the last page regardless of the token. Default pageSize is 25, so return 25 rows.
    state.pages = {
      '': { traces: makeTraces(25, 'p1'), next_page_token: 'token-2' },
      'token-2': { traces: makeTraces(3, 'p2'), next_page_token: undefined },
    };
    renderPage();
    expect(await findTraceRow('p1-000')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Next page' }));
    expect(await findTraceRow('p2-000')).toBeInTheDocument();
    // The second search carried the recorded next-page token.
    expect(state.searchCalls.some((c) => c.page_token === 'token-2')).toBe(true);

    const callsBeforeBack = state.searchCalls.length;
    await user.click(screen.getByRole('button', { name: 'Previous page' }));
    expect(await findTraceRow('p1-000')).toBeInTheDocument();
    // Going back to a cached page issues no new request.
    expect(state.searchCalls.length).toBe(callsBeforeBack);
    // Two full renders of the heavy shared table (Next then Prev) push this past the 5s default in
    // jsdom; a per-test timeout keeps it green without the lint-forbidden global jest.setTimeout.
  }, 20000);

  test('changing page size sends max_results and resets to page 1', async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    state.pages = { '': { traces: makeTraces(3), next_page_token: 'token-2' } };
    renderPage();
    expect(await findTraceRow('tr-000')).toBeInTheDocument();
    expect(state.searchCalls[0].max_results).toBe(25);

    await user.click(screen.getByLabelText('Rows per page'));
    await user.click(await screen.findByRole('option', { name: '100' }));

    await waitFor(() => expect(state.searchCalls.some((c) => c.max_results === 100)).toBe(true));
    expect(new URLSearchParams(lastSearch).get('pageSize')).toBe('100');
    expect(new URLSearchParams(lastSearch).get('page')).toBeNull();
    // Re-render on the page-size change is slow in jsdom; per-test timeout avoids global jest.setTimeout.
  }, 20000);

  test('sorting by Time sends an order_by and resets the page', async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderPage();
    expect(await findTraceRow('tr-000')).toBeInTheDocument();

    // The sortable start-time header (relabeled "Time") renders a button; the date-range selector
    // with the same label is a combobox, so this targets the column header unambiguously.
    await user.click(screen.getByRole('button', { name: /^Time$/ }));
    await waitFor(() => expect(state.searchCalls.some((c) => c.order_by?.[0]?.startsWith('timestamp'))).toBe(true));
  });

  test('non-sortable headers (State) expose no sort control', async () => {
    renderPage();
    await findTraceRow('tr-000');
    // Sortable headers render a button; the display-only State column must not.
    expect(screen.queryByRole('button', { name: /^State$/ })).not.toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'State' })).toBeInTheDocument();
  });

  test('select rows → Actions (2) → Delete is enabled in OSS and opens the confirm modal', async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderPage();
    await findTraceRow('tr-000');

    await user.click(screen.getByRole('checkbox', { name: 'Select trace tr-000' }));
    await user.click(screen.getByRole('checkbox', { name: 'Select trace tr-001' }));

    // The Actions button's accessible name is its aria-label; the "(2)" count is visible text.
    const actionsButton = await screen.findByRole('button', { name: 'Actions for selected traces' });
    expect(actionsButton).toHaveTextContent('Actions (2)');
    await user.click(actionsButton);

    // OSS traces have no UC gate (isDeleteDisabled === false), so Delete is enabled and clicking it
    // opens the delete-confirmation modal (mirrors the datasets-v2 delete flow).
    const deleteItem = await screen.findByRole('menuitem', { name: /Delete/ });
    expect(deleteItem).not.toHaveAttribute('aria-disabled', 'true');
    await user.click(deleteItem);
    expect(await screen.findByRole('dialog')).toHaveTextContent(/Delete/);
  }, 20000);

  test('searching commits only on Enter, not per keystroke, then sends an ILIKE filter', async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderPage();
    await findTraceRow('tr-000');

    // Type key-by-key (the sanctioned escape hatch for the no-userevent-type lint rule). Typing alone
    // must NOT fire the search — the box commits on Enter now, not per keystroke.
    const searchBox = screen.getByPlaceholderText('Search traces by id, input, or output');
    await slowlyTypeEachKey(searchBox, 'hello', { user });
    expect(state.searchCalls.some((c) => c.filter?.includes("trace.text ILIKE '%hello%'"))).toBe(false);

    // Pressing Enter commits the typed value and the ILIKE search fires.
    pressEnter(searchBox);
    await waitFor(() =>
      expect(state.searchCalls.some((c) => c.filter?.includes("trace.text ILIKE '%hello%'"))).toBe(true),
    );
  });

  test('clearing the search (X) commits an empty search immediately', async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderPage();
    await findTraceRow('tr-000');

    // Commit a search first (Enter), then clear it.
    const searchBox = screen.getByPlaceholderText('Search traces by id, input, or output');
    await slowlyTypeEachKey(searchBox, 'hello', { user });
    pressEnter(searchBox);
    await waitFor(() => expect(new URLSearchParams(lastSearch).get('q')).toBe('hello'));

    // Clicking the input's clear (X) affordance commits the empty search right away (no Enter needed).
    // Du Bois Input renders the X with the accessible name `close-circle` (matches the datasets-v2 tests).
    await user.click(screen.getByLabelText('close-circle'));
    await waitFor(() => expect(new URLSearchParams(lastSearch).get('q')).toBeNull());
  });

  // TODO(traces-v4): The OSS empty state renders TracesViewTableNoTracesQuickstart; this test asserts the Databricks TracingQuickStart CTA copy. Rewrite for the OSS quickstart.

  test.skip('shows the tracing quickstart CTA when the experiment has no traces', async () => {
    state.pages = { '': { traces: [], next_page_token: undefined } };
    renderPage();
    // The unmocked experiment query surfaces no kind, so the non-GenAI generic quickstart renders
    // ("No traces recorded") rather than the old generic shared "No traces yet" empty state.
    expect(await screen.findByText('No traces recorded')).toBeInTheDocument();
    expect(screen.queryByText('No traces yet')).not.toBeInTheDocument();
  });

  describe('robustness', () => {
    test('a search error shows an error state with a working Retry', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      state.searchShouldFail = true;
      renderPage();
      expect(await screen.findByText("Couldn't load traces")).toBeInTheDocument();

      // Recover: next fetch succeeds.
      state.searchShouldFail = false;
      await user.click(screen.getByRole('button', { name: 'Retry' }));
      expect(await findTraceRow('tr-000')).toBeInTheDocument();
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('a first-load error surfaces the server message (not the generic fallback)', async () => {
      // A permission error must read distinctly from a timeout: the raw server message is shown so the
      // cause isn't discarded (the reported regression vs v3, where every error looked identical).
      state.searchShouldFail = true;
      state.searchErrorMessage = 'PERMISSION_DENIED: user lacks access to the traces table';
      renderPage();
      expect(await screen.findByText("Couldn't load traces")).toBeInTheDocument();
      expect(screen.getByText(/PERMISSION_DENIED: user lacks access to the traces table/)).toBeInTheDocument();
    });

    test('a first-load SQL-warehouse timeout shows the larger-warehouse hint', async () => {
      // A SQL-warehouse timeout is detected (via the shared isSqlWarehouseTimeoutError) and gets the
      // actionable "try a larger warehouse" hint, matching v3 — not the raw message or generic text.
      state.searchShouldFail = true;
      state.searchErrorMessage = 'Timeout while issuing SQL query to fetch traces';
      renderPage();
      expect(await screen.findByText("Couldn't load traces")).toBeInTheDocument();
      expect(screen.getByText(/try selecting a larger SQL warehouse/)).toBeInTheDocument();
    });

    test('a malformed trace row (missing assessments, unparseable metadata) renders without throwing', async () => {
      state.pages = {
        '': {
          traces: [
            makeTrace('tr-bad', { assessments: undefined, trace_metadata: { 'mlflow.trace.tokenUsage': 'not json' } }),
          ],
          next_page_token: undefined,
        },
      };
      renderPage();
      // Row still renders; the bad token metadata simply shows nothing rather than crashing.
      expect(await findTraceRow('tr-bad')).toBeInTheDocument();
    });
  });

  describe('layout stability', () => {
    test('first load renders the real header plus exactly pageSize skeleton rows', async () => {
      // Delay the search so the skeleton is observable.
      let resolveSearch: (() => void) | undefined;
      server.use(
        rest.post(SEARCH_ENDPOINT, async (_req, res, ctx) => {
          await new Promise<void>((r) => {
            resolveSearch = r;
          });
          return res(ctx.json({ traces: makeTraces(3), next_page_token: undefined }));
        }),
      );
      renderPage();
      // Header is real from the first paint.
      expect(await screen.findByRole('columnheader', { name: 'Time' })).toBeInTheDocument();
      // Skeleton status region is present during first load.
      expect(screen.getByRole('region', { name: 'Traces' })).toHaveAttribute('aria-busy', 'true');
      resolveSearch?.();
      expect(await findTraceRow('tr-000')).toBeInTheDocument();
    });

    test('reloading shows the skeleton again (keyed off isFetching, not just first load)', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      expect(await findTraceRow('tr-000')).toBeInTheDocument();

      // Make the *next* search hang so the reload's in-flight state is observable. A sort change
      // reuses keepPreviousData (isLoading stays false, isFetching flips true) — the exact reload
      // scenario that must now render the skeleton rather than keep the stale rows.
      let resolveReload: (() => void) | undefined;
      server.use(
        rest.post(SEARCH_ENDPOINT, async (req, res, ctx) => {
          const body = (await req.json()) as SearchCall;
          state.searchCalls.push(body);
          await new Promise<void>((r) => {
            resolveReload = r;
          });
          return res(ctx.json({ traces: makeTraces(3), next_page_token: undefined }));
        }),
      );

      await user.click(screen.getByRole('button', { name: /^Time$/ }));

      // Skeleton is back and the prior rows are replaced by it (rather than kept via keepPreviousData).
      await waitFor(() => expect(screen.getByRole('region', { name: 'Traces' })).toHaveAttribute('aria-busy', 'true'));
      expect(queryTraceRow('tr-000')).not.toBeInTheDocument();

      resolveReload?.();
      expect(await findTraceRow('tr-000')).toBeInTheDocument();
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load
  });

  describe('opening a trace', () => {
    test('clicking a UC-backed row opens the drawer with a V4 long identifier (not a bare hex id)', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      await user.click(await findTraceRow('tr-000'));

      // The mock trace_location is UC_SCHEMA cat.sch, so the URL must carry the V4 long id
      // `trace:/cat.sch/tr-000` — the bare hex id would hit the legacy path and be rejected as
      // "Invalid request id". The URL long id also keeps the row-selection highlight matching.
      const longId = 'trace:/cat.sch/tr-000';
      const drawer = await screen.findByRole('dialog');
      expect(drawer).toBeInTheDocument();
      expect(new URLSearchParams(lastSearch).get('traceId')).toBe(longId);
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('the drawer header shows the trace input preview, not the raw id (matches v3)', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      await user.click(await findTraceRow('tr-000'));

      // v3-faithful: the header title heading is the input preview ("request for tr-000"), not the
      // long id `trace:/cat.sch/tr-000` that the drawer previously showed.
      const drawer = await screen.findByRole('dialog');
      expect(await within(drawer).findByRole('heading', { name: 'request for tr-000' })).toBeInTheDocument();
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load
  });

  describe('session column', () => {
    test('is visible by default when the page has session-tagged traces', async () => {
      state.pages = { '': { traces: [makeSessionTrace('tr-000')], next_page_token: undefined } };
      renderPage();
      await findTraceRow('tr-000');
      expect(screen.getByRole('columnheader', { name: 'Session' })).toBeInTheDocument();
    });

    test('is hidden by default when no trace on the page has a session', async () => {
      renderPage();
      await findTraceRow('tr-000');
      expect(screen.queryByRole('columnheader', { name: 'Session' })).not.toBeInTheDocument();
    });

    test('an explicit toggle sticks even on a page without sessions', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      const { unmount } = renderPage();
      await findTraceRow('tr-000');

      // Turn Session on via the column selector, even though this page has no sessions.
      await user.click(screen.getByRole('button', { name: 'Select visible columns' }));
      await user.click(await screen.findByRole('menuitemcheckbox', { name: 'Session' }));
      expect(await screen.findByRole('columnheader', { name: 'Session' })).toBeInTheDocument();

      // Remount: the explicit "on" choice persists (a sticky override, not the data-driven default).
      unmount();
      renderPage();
      expect(await screen.findByRole('columnheader', { name: 'Session' })).toBeInTheDocument();
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('the session value links to the chat-session page with the trace deep-linked', async () => {
      state.pages = { '': { traces: [makeSessionTrace('tr-000', 'sess-1')], next_page_token: undefined } };
      renderPage();
      await findTraceRow('tr-000');

      // The session tag is wrapped in a link to the single-chat-session route (like v1), carrying the
      // current trace via ?selectedTraceId so the destination opens on that trace.
      const link = screen.getByRole('link', { name: 'sess-1' });
      const href = link.getAttribute('href') ?? '';
      expect(href).toContain('/experiments/exp-1/chat-sessions/sess-1');
      expect(href).toContain('selectedTraceId=tr-000');
    });
  });

  describe('assessment columns', () => {
    const traceWithAssessment = makeTrace('tr-000', { assessments: [makeFeedbackAssessment('relevance', 'yes')] });

    test('renders a column per on-page assessment with its value tag', async () => {
      state.pages = { '': { traces: [traceWithAssessment], next_page_token: undefined } };
      renderPage();
      await findTraceRow('tr-000');

      expect(screen.getByRole('columnheader', { name: 'relevance' })).toBeInTheDocument();
      // 'yes' feedback renders as a "Yes" tag (AssessmentDisplayValue).
      expect(screen.getByText('Yes')).toBeInTheDocument();
    });

    test('toggling an assessment off hides its column and the choice persists', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      state.pages = { '': { traces: [traceWithAssessment], next_page_token: undefined } };
      const { unmount } = renderPage();
      await findTraceRow('tr-000');
      expect(screen.getByRole('columnheader', { name: 'relevance' })).toBeInTheDocument();

      await user.click(screen.getByRole('button', { name: 'Select visible columns' }));
      await user.click(await screen.findByRole('menuitemcheckbox', { name: 'relevance' }));
      await waitFor(() => expect(screen.queryByRole('columnheader', { name: 'relevance' })).not.toBeInTheDocument());

      // Remount: the opt-out persists per experiment even though the page still has the assessment.
      unmount();
      renderPage();
      await findTraceRow('tr-000');
      expect(screen.queryByRole('columnheader', { name: 'relevance' })).not.toBeInTheDocument();
      // Two full page renders (mount + remount) under parallel jsdom load — bump off the 5s default.
    }, 20000);
  });

  describe('tags column', () => {
    test('is visible by default and shows user tags with a "+N" preview', async () => {
      state.pages = {
        '': { traces: [makeTaggedTrace('tr-000', { env: 'prod', team: 'ml' })], next_page_token: undefined },
      };
      renderPage();
      await findTraceRow('tr-000');

      expect(screen.getByRole('columnheader', { name: 'Tags' })).toBeInTheDocument();
      // First tag is shown as a pill; the second is collapsed into "+1".
      expect(screen.getByText('env: prod')).toBeInTheDocument();
      expect(screen.getByText('+1')).toBeInTheDocument();
      // Heavy full-page render is slow under parallel jsdom load; per-test timeout avoids the
      // lint-forbidden global jest.setTimeout.
    }, 20000);

    test('hides internal mlflow.* tags from the preview', async () => {
      // Only an internal tag → nothing user-facing → renders the "-" empty value.
      state.pages = {
        '': {
          traces: [makeTrace('tr-000', { tags: { 'mlflow.trace.sizeBytes': '123' } })],
          next_page_token: undefined,
        },
      };
      renderPage();
      await findTraceRow('tr-000');

      expect(screen.getByRole('columnheader', { name: 'Tags' })).toBeInTheDocument();
      expect(screen.queryByText(/mlflow\.trace\.sizeBytes/)).not.toBeInTheDocument();
    });
  });

  describe('filter by tag', () => {
    test('clicking the first tag pill applies a tags clause and writes the tag URL param', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      state.pages = {
        '': { traces: [makeTaggedTrace('tr-000', { env: 'prod', team: 'ml' })], next_page_token: undefined },
      };
      renderPage();
      await findTraceRow('tr-000');

      await user.click(screen.getByRole('button', { name: 'Filter by tag env: prod' }));

      // The refetch carries the compiled tag clause…
      await waitFor(() => expect(state.searchCalls.some((c) => c.filter?.includes("tags.env = 'prod'"))).toBe(true));
      // …and the filter is persisted in the URL as a repeatable, encoded `tag` param.
      expect(new URLSearchParams(lastSearch).getAll('tag')).toContain('env=prod');
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('clicking a tag in the overflow hover card applies that tag (not just the first pill)', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      state.pages = {
        '': { traces: [makeTaggedTrace('tr-000', { env: 'prod', team: 'ml' })], next_page_token: undefined },
      };
      renderPage();
      await findTraceRow('tr-000');

      // Hover the tags cell's "+N" activator (the hover-card trigger) to reveal the full tag list,
      // then click the second tag's filter pill — the `team: ml` pill lives only in the overflow list
      // (the inline pill is the first tag, `env: prod`), so finding it proves the hover card opened
      // with its content visible. `fireEvent.click` (not `user.click`) is deliberate: userEvent moves
      // the pointer, which hovers out of the trigger and dismisses the card before the click lands.
      await user.hover(screen.getByRole('button', { name: 'Open trace tr-000 — tags' }));
      fireEvent.click(await screen.findByRole('button', { name: 'Filter by tag team: ml' }));

      await waitFor(() => expect(state.searchCalls.some((c) => c.filter?.includes("tags.team = 'ml'"))).toBe(true));
      expect(new URLSearchParams(lastSearch).getAll('tag')).toContain('team=ml');
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('clicking a tag pill filters but does NOT open the trace drawer', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      state.pages = {
        '': { traces: [makeTaggedTrace('tr-000', { env: 'prod', team: 'ml' })], next_page_token: undefined },
      };
      renderPage();
      await findTraceRow('tr-000');

      await user.click(screen.getByRole('button', { name: 'Filter by tag env: prod' }));

      // The filter applied (URL param written)…
      await waitFor(() => expect(new URLSearchParams(lastSearch).getAll('tag')).toContain('env=prod'));
      // …but the drawer never opened and no traceId was set (stopPropagation kept it off the row click).
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
      expect(new URLSearchParams(lastSearch).get('traceId')).toBeNull();
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('a deep-linked tag param drives the initial request filter', async () => {
      state.pages = { '': { traces: makeTraces(3), next_page_token: undefined } };
      renderPage({ initialUrl: `${URL}?tag=env%3Dprod` });
      await findTraceRow('tr-000');

      expect(state.searchCalls[0]?.filter).toContain("tags.env = 'prod'");
    });
  });

  describe('resizable columns', () => {
    test('applies a persisted per-column width on mount', async () => {
      // A width persisted from a prior session (by the resize handle → useLocalStorage) must be
      // read back and applied to the column on the next mount. Seed it via the same scoped/versioned
      // key shape the hook writes, then assert a default-visible header renders at that width. Input
      // is a growing column, so its persisted width becomes the flex-basis (maxWidth is unset so the
      // cap can't block growth).
      const sizesKey = `${TRACE_COLUMN_SIZES_STORAGE_KEY_PREFIX}.${EXPERIMENT_ID}`;
      setLocalStorageItem(sizesKey, COLUMN_SIZES_STORAGE_VERSION, true, { input: 321 });

      renderPage();
      const header = await screen.findByRole('columnheader', { name: 'Input' });
      expect(header).toHaveStyle({ flexBasis: '321px' });
    });
  });

  describe('state column', () => {
    test.each([
      { state: 'OK' as const, label: 'OK' },
      { state: 'ERROR' as const, label: 'Error' },
      { state: 'IN_PROGRESS' as const, label: 'In progress' },
    ])('renders the $label badge for a $state trace', async ({ state: traceState, label }) => {
      state.pages = { '': { traces: [makeTrace('tr-000', { state: traceState })], next_page_token: undefined } };
      renderPage();
      await findTraceRow('tr-000');
      expect(screen.getByText(label)).toBeInTheDocument();
    });

    test('renders "-" (no badge) for a STATE_UNSPECIFIED trace', async () => {
      state.pages = {
        '': { traces: [makeTrace('tr-000', { state: 'STATE_UNSPECIFIED' })], next_page_token: undefined },
      };
      renderPage();
      await findTraceRow('tr-000');
      // None of the state labels render for an unspecified state.
      expect(screen.queryByText('OK')).not.toBeInTheDocument();
      expect(screen.queryByText('Error')).not.toBeInTheDocument();
      expect(screen.queryByText('In progress')).not.toBeInTheDocument();
    });
  });

  describe('pagination bar', () => {
    test('is shown with Prev/Next disabled when there is only a single page of results', async () => {
      // No next_page_token means a single page: the bar still renders (page-size selector lives in
      // it) and both cursor buttons are disabled (Next via the last-page marker, Prev on page 1).
      state.pages = { '': { traces: makeTraces(3), next_page_token: undefined } };
      renderPage();
      await findTraceRow('tr-000');

      // Page-size selector is present…
      expect(screen.getByLabelText('Rows per page')).toBeInTheDocument();
      // …and Next is disabled on a single page (no next token), as is Prev on page 1.
      expect(screen.getByRole('button', { name: 'Next page' })).toBeDisabled();
      expect(screen.getByRole('button', { name: 'Previous page' })).toBeDisabled();
    });

    test('a partial final page disables Next even when the server sends a next_page_token', async () => {
      // Reproduces the reported bug: the long-running backend returns a real token on every non-empty
      // page including a partial final one. 21 rows at the default page size 25 is a short page, so the
      // frontend must treat it as the last page and disable Next — despite the non-empty token.
      state.pages = { '': { traces: makeTraces(21), next_page_token: 'phantom-token' } };
      renderPage();
      await findTraceRow('tr-000');

      expect(screen.getByRole('button', { name: 'Next page' })).toBeDisabled();
      expect(screen.getByRole('button', { name: 'Previous page' })).toBeDisabled();
      // Heavy full-page render is slow under parallel jsdom load; per-test timeout avoids the
      // lint-forbidden global jest.setTimeout (matches the sibling pagination tests).
    }, 20000);

    test('an exactly-full last page → Next enabled; clicking it shows "No more results" with the bar still present', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      // Page 1 is exactly full (25 rows at pageSize 25) with a real token, so the client can't yet know
      // page 2 is empty — Next must stay enabled. Clicking Next fetches an empty page 2.
      state.pages = {
        '': { traces: makeTraces(25, 'p1'), next_page_token: 'token-2' },
        'token-2': { traces: [], next_page_token: undefined },
      };
      renderPage();
      expect(await findTraceRow('p1-000')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Next page' })).toBeEnabled();

      await user.click(screen.getByRole('button', { name: 'Next page' }));

      // The distinct end-of-results state renders — NOT the initial "No traces yet" empty state…
      expect(await screen.findByText('No more results')).toBeInTheDocument();
      expect(screen.queryByText('No traces yet')).not.toBeInTheDocument();
      // …the pagination bar is still present (page-size selector lives in it)…
      expect(screen.getByLabelText('Rows per page')).toBeInTheDocument();
      // …Prev is enabled (back to page 1), Next is disabled (the empty page is terminal).
      expect(screen.getByRole('button', { name: 'Previous page' })).toBeEnabled();
      expect(screen.getByRole('button', { name: 'Next page' })).toBeDisabled();

      // Stepping back returns to page 1's rows (served from cache, no refetch needed).
      await user.click(screen.getByRole('button', { name: 'Previous page' }));
      expect(await findTraceRow('p1-000')).toBeInTheDocument();
      // Full-page renders + two paginations are slow under parallel jsdom load; per-test timeout
      // avoids the lint-forbidden global jest.setTimeout.
    }, 20000);

    test('shows the "{n} of {total}" count — current page rows out of the metrics total', async () => {
      // 3 rows on the page; the trace-metrics endpoint reports 42 total.
      state.pages = { '': { traces: makeTraces(3), next_page_token: undefined } };
      metricsTotalCount = 42;
      renderPage();
      await findTraceRow('tr-000');

      expect(await screen.findByText('3 of 42')).toBeInTheDocument();
    }, 20000);
  });

  describe('OSS data path', () => {
    test('OSS always uses MLFLOW_EXPERIMENT location regardless of storageUCSchema parameter', async () => {
      renderPage(); // renderPage uses STORAGE_UC_SCHEMA = 'cat.sch', but OSS ignores it
      await findTraceRow('tr-000');

      const location = state.searchCalls[0]?.locations?.[0];
      // OSS always uses MLFLOW_EXPERIMENT; the storageUCSchema parameter is for Databricks only
      expect(location?.type).toBe('MLFLOW_EXPERIMENT');
      expect(location?.mlflow_experiment).toEqual({ experiment_id: EXPERIMENT_ID });
    });
  });

  describe('filter', () => {
    const applyErrorStateClause = async (user: ReturnType<typeof userEvent.setup>) => {
      // Open the popover; the first row defaults to Field=State, Operator=`=`. Pick a state value.
      await user.click(screen.getByRole('button', { name: /Filters/ }));
      await user.click(await screen.findByLabelText('Filter value'));
      await user.click(await screen.findByRole('option', { name: 'Error' }));
      await user.click(screen.getByRole('button', { name: 'Apply filters' }));
      await waitFor(() =>
        expect(state.searchCalls.some((c) => c.filter?.includes("attributes.status = 'ERROR'"))).toBe(true),
      );
    };

    // Serve a filter-specific row set so a clear is observable in the DOM: an ERROR-filtered search
    // returns `err-000`, an unfiltered search returns `all-000`. (The default handler serves the same
    // rows regardless of filter, and React Query serves the unfiltered page from cache on clear — so
    // asserting on network calls can't distinguish "filter cleared" from "filter still applied".)
    const useFilterDistinctRows = () => {
      server.use(
        rest.post(SEARCH_ENDPOINT, async (req, res, ctx) => {
          const body = (await req.json()) as SearchCall;
          state.searchCalls.push(body);
          const hasError = body.filter?.includes("attributes.status = 'ERROR'");
          const id = hasError ? 'err-000' : 'all-000';
          return res(
            ctx.json({
              traces: [makeTrace(id, { state: hasError ? 'ERROR' : 'OK' })],
              next_page_token: undefined,
            }),
          );
        }),
      );
    };

    test('building a State clause and applying it sends the compiled status filter', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      await findTraceRow('tr-000');

      await applyErrorStateClause(user);
      expect(state.searchCalls.some((c) => c.filter?.includes("attributes.status = 'ERROR'"))).toBe(true);
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    // TODO(traces-v4): service_name filtering was dropped in OSS (the SearchTracesV3 parser rejects span.service_name). Field + clause removed by design.

    test.skip('building a service name clause and applying it sends the compiled span.service_name filter', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      await findTraceRow('tr-000');

      // Open the popover, change filter field to service name, type a value, and apply.
      await user.click(screen.getByRole('button', { name: /Filters/ }));
      await user.click(await screen.findByLabelText('Filter field'));
      await user.click(await screen.findByRole('option', { name: 'Service name' }));
      await slowlyTypeEachKey(await screen.findByLabelText('Filter value'), 'my-service', { user });
      await user.click(screen.getByRole('button', { name: 'Apply filters' }));

      await waitFor(() =>
        expect(state.searchCalls.some((c) => c.filter?.includes("span.service_name = 'my-service'"))).toBe(true),
      );
    });

    test('the clear-all button clears an applied clause and the unfiltered rows return', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      useFilterDistinctRows();
      renderPage();
      expect(await findTraceRow('all-000')).toBeInTheDocument();

      await applyErrorStateClause(user);
      // The filtered result set is shown and the count badge lights up.
      expect(await findTraceRow('err-000')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Filters/ })).toHaveTextContent('(1)');

      // Clear-all is a real, keyboard-reachable button (a sibling of the trigger, not a nested icon),
      // so it's discoverable by role/name — clicking it drops the clause, its badge, and the filter.
      await user.click(screen.getByRole('button', { name: 'Clear all filters' }));

      expect(await findTraceRow('all-000')).toBeInTheDocument();
      expect(queryTraceRow('err-000')).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Filters/ })).not.toHaveTextContent('(1)');
      // Multi-step filter interaction on a full page render — bump off the flaky 5s default.
    }, 20000);

    test('the clear-all button also clears an applied click-to-filter tag', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      state.pages = {
        '': { traces: [makeTaggedTrace('tr-000', { env: 'prod', team: 'ml' })], next_page_token: undefined },
      };
      renderPage();
      await findTraceRow('tr-000');

      // A tag filter (a URL-backed concept, distinct from the popover clauses) counts toward the badge.
      await user.click(screen.getByRole('button', { name: 'Filter by tag env: prod' }));
      await waitFor(() => expect(new URLSearchParams(lastSearch).getAll('tag')).toContain('env=prod'));
      expect(screen.getByRole('button', { name: /Filters/ })).toHaveTextContent('(1)');

      // Clear-all must clear the tag param too (not only the popover clauses), or the badge would stay
      // lit and the filter would remain applied — the reported bug.
      await user.click(screen.getByRole('button', { name: 'Clear all filters' }));

      expect(new URLSearchParams(lastSearch).getAll('tag')).not.toContain('env=prod');
      expect(screen.getByRole('button', { name: /Filters/ })).not.toHaveTextContent('(1)');
      // Multi-step tag-filter interaction on a full page render — bump off the flaky 5s default.
    }, 20000);

    test('the popover Clear filters button clears the applied clause and the unfiltered rows return', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      useFilterDistinctRows();
      renderPage();
      expect(await findTraceRow('all-000')).toBeInTheDocument();

      await applyErrorStateClause(user);
      expect(await findTraceRow('err-000')).toBeInTheDocument();

      // Reopen the popover and use its Clear button (distinct from removing a single clause row).
      await user.click(screen.getByRole('button', { name: /Filters/ }));
      await user.click(await screen.findByRole('button', { name: 'Clear filters' }));

      expect(await findTraceRow('all-000')).toBeInTheDocument();
      expect(queryTraceRow('err-000')).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Filters/ })).not.toHaveTextContent('(1)');
      // Filter apply + clear each re-render the heavy table; per-test timeout avoids the
      // lint-forbidden global jest.setTimeout under parallel jsdom load.
    }, 20000);
  });

  describe('default time range', () => {
    test('displays the 15 minute range on mount when the URL has no startTimeLabel', async () => {
      renderPage();
      await findTraceRow('tr-000');
      // The standalone hook resolves the default without writing it to the URL, so assert the
      // dropdown *displays* the default rather than checking the URL param.
      expect(screen.getByRole('button', { name: 'Time range: Last 15 minutes' })).toHaveTextContent(
        /^Last 15 minutes$/,
      );
    });

    test('ignores a v3 saved time selection (isolated v4 localStorage key)', async () => {
      // Seed the legacy v3 key exactly as the shared useMonitoringFilters persists it (version 1,
      // scoped). v4 uses a distinct key, so it must not read this — the dropdown should still show the
      // v4 default.
      setLocalStorageItem(`traces_useMonitoringFilters_${EXPERIMENT_ID}`, 1, true, { startTimeLabel: 'LAST_30_DAYS' });
      renderPage();
      await findTraceRow('tr-000');
      expect(screen.getByRole('button', { name: 'Time range: Last 15 minutes' })).toHaveTextContent(
        /^Last 15 minutes$/,
      );
    });
  });

  describe('URL preservation (v3-compatible params)', () => {
    test('a full deep-link drives the correct initial request and the params survive on the URL', async () => {
      state.pages = {
        // page=2 needs a reachable cursor; but the token cache is memory-only, so on a fresh load the
        // stale-page recovery resets to page 1 (documented behavior). The request assertions below
        // target the filter/order_by/max_results the URL drives, which are page-independent.
        '': { traces: makeTraces(3), next_page_token: undefined },
      };
      renderPage({
        initialUrl: `${URL}?q=hello&pageSize=100&sort=duration&dir=asc&startTimeLabel=LAST_7_DAYS&tag=env%3Dprod`,
      });
      await findTraceRow('tr-000');

      const firstCall = state.searchCalls[0];
      // Search → ILIKE; tag → compiled tags clause; time-range label → ms bounds on timestamp_ms.
      expect(firstCall.filter).toContain("trace.text ILIKE '%hello%'");
      expect(firstCall.filter).toContain("tags.env = 'prod'");
      expect(firstCall.filter).toContain('attributes.timestamp_ms >');
      expect(firstCall.filter).toContain('attributes.timestamp_ms <');
      // Sort → order_by; pageSize → max_results.
      expect(firstCall.order_by?.[0]).toBe('execution_time ASC');
      expect(firstCall.max_results).toBe(100);

      // The shared/v4 params survive on the URL after mount.
      const search = new URLSearchParams(lastSearch);
      expect(search.get('q')).toBe('hello');
      expect(search.get('pageSize')).toBe('100');
      expect(search.get('sort')).toBe('duration');
      expect(search.get('dir')).toBe('asc');
      expect(search.get('startTimeLabel')).toBe('LAST_7_DAYS');
      expect(search.getAll('tag')).toContain('env=prod');
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('the shared v3 CUSTOM time bounds are honored on mount and preserved across an in-tab action', async () => {
      const start = '2025-01-01T00:00:00.000Z';
      const end = '2025-01-02T00:00:00.000Z';
      const user = userEvent.setup();
      renderPage({
        initialUrl: `${URL}?startTimeLabel=CUSTOM&startTime=${encodeURIComponent(start)}&endTime=${encodeURIComponent(
          end,
        )}`,
      });
      await findTraceRow('tr-000');

      // The explicit CUSTOM bounds map to the ms timestamps on the initial request.
      const startMs = new Date(start).getTime();
      const endMs = new Date(end).getTime();
      expect(state.searchCalls[0].filter).toContain(`attributes.timestamp_ms > ${startMs}`);
      expect(state.searchCalls[0].filter).toContain(`attributes.timestamp_ms < ${endMs}`);

      // An in-tab action (search) issues a new request that still carries the same time bounds…
      // The search commits on Enter, so type then press Enter.
      const searchBox = screen.getByPlaceholderText('Search traces by id, input, or output');
      await slowlyTypeEachKey(searchBox, 'x', { user });
      pressEnter(searchBox);
      await waitFor(() =>
        expect(
          state.searchCalls.some(
            (c) => c.filter?.includes(`attributes.timestamp_ms > ${startMs}`) && c.filter?.includes("ILIKE '%x%'"),
          ),
        ).toBe(true),
      );
      // …and the shared v3 time params are preserved on the URL.
      const search = new URLSearchParams(lastSearch);
      expect(search.get('startTimeLabel')).toBe('CUSTOM');
      expect(search.get('startTime')).toBe(start);
      expect(search.get('endTime')).toBe(end);
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('a time-range change issues a new request with new time bounds', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      await findTraceRow('tr-000');
      const callsBefore = state.searchCalls.length;

      // Open the time-range dropdown and pick a different preset.
      await user.click(screen.getByRole('button', { name: 'Time range: Last 15 minutes' }));
      await user.click(await screen.findByRole('option', { name: 'Last 7 days' }));

      // A new search fires (the filter's time bounds changed), so the call count grows.
      await waitFor(() => expect(state.searchCalls.length).toBeGreaterThan(callsBefore));
      expect(new URLSearchParams(lastSearch).get('startTimeLabel')).toBe('LAST_7_DAYS');
      expect(screen.getByRole('button', { name: 'Time range: Last 7 days' })).toHaveTextContent(/^Last 7 days$/);
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('selecting a custom value shows the absolute range picker and preset button', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      await findTraceRow('tr-000');

      await user.click(screen.getByRole('button', { name: 'Time range: Last 15 minutes' }));
      await user.click(await screen.findByRole('option', { name: 'Custom' }));

      await waitFor(() => expect(new URLSearchParams(lastSearch).get('startTimeLabel')).toBe('CUSTOM'));
      expect(screen.getAllByRole('textbox', { name: 'Select Date and Time' })).toHaveLength(2);
      expect(screen.getByRole('button', { name: 'Time range' })).toBeInTheDocument();

      // The calendar button keeps the preset choices (without a redundant Custom option) and
      // returns to the compact preset control after a selection.
      await user.click(screen.getByRole('button', { name: 'Time range' }));
      expect(screen.queryByRole('option', { name: 'Custom' })).not.toBeInTheDocument();
      await user.click(await screen.findByRole('option', { name: 'Last 7 days' }));
      expect(screen.getByRole('button', { name: 'Time range: Last 7 days' })).toBeInTheDocument();
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load
  });

  describe('trace actions ("Use for evaluation" group)', () => {
    // Opens the Actions dropdown after selecting the first trace, returning the menu for scoping.
    const openActionsMenuForFirstTrace = async (user: ReturnType<typeof userEvent.setup>) => {
      await findTraceRow('tr-000');
      await user.click(screen.getByRole('checkbox', { name: 'Select trace tr-000' }));
      await user.click(await screen.findByRole('button', { name: 'Actions for selected traces' }));
      return screen.findByRole('menu');
    };

    test('always offers "Run scorers" and "Add to evaluation dataset" for a selection', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      const menu = await openActionsMenuForFirstTrace(user);

      // These two are always available (no extra feature gate) — the core evaluation actions.
      expect(within(menu).getByRole('menuitem', { name: 'Run scorers' })).toBeInTheDocument();
      expect(within(menu).getByRole('menuitem', { name: 'Add to evaluation dataset' })).toBeInTheDocument();
      // Delete remains, below the evaluation group. In OSS traces have no UC gate, so it's enabled.
      const deleteItem = within(menu).getByRole('menuitem', { name: /Delete/ });
      expect(deleteItem).toBeInTheDocument();
      expect(deleteItem).not.toHaveAttribute('aria-disabled', 'true');
    }, 20000);

    test('offers "Flag for review" — review queues ship in OSS, matching v3', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      const menu = await openActionsMenuForFirstTrace(user);

      // v3 (TracesV3Logs) wires AddToReviewQueueDropdown with addToReviewQueue: true in OSS, and OSS
      // ships a full review-queue page/route, so "Flag for review" must be offered in v4 too.
      expect(within(menu).getByRole('menuitem', { name: 'Flag for review' })).toBeInTheDocument();
    }, 20000);

    test('opening "Run scorers" launches the scorer-selection modal for the selected trace', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      const menu = await openActionsMenuForFirstTrace(user);
      await user.click(within(menu).getByRole('menuitem', { name: 'Run scorers' }));

      // The shared run-judges modal opens scoped to the single selected trace. (The menu item reads
      // "Run scorers"; the modal title uses the underlying "judge" terminology.)
      expect(await screen.findByRole('dialog', { name: /Run judge on trace/ })).toBeInTheDocument();
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('bulk actions run on the full cross-page selection (select on page 1, page to 2, Run scorers)', async () => {
      // The selection now stores each trace's full info keyed by id, so it spans pages. Selecting one
      // trace on page 1, paging to page 2, selecting another there, then running scorers must scope the
      // action to BOTH selected traces — not just the page-2 subset (the reported "Run scorers (2)" runs
      // on 0/1 traces bug).
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      state.pages = {
        '': { traces: makeTraces(50, 'p1'), next_page_token: 'token-2' },
        'token-2': { traces: makeTraces(50, 'p2'), next_page_token: undefined },
      };
      renderPage();
      expect(await findTraceRow('p1-000')).toBeInTheDocument();

      // Select a trace on page 1, then paginate to page 2 (the selection persists across the page swap).
      await user.click(screen.getByRole('checkbox', { name: 'Select trace p1-000' }));
      await user.click(screen.getByRole('button', { name: 'Next page' }));
      expect(await findTraceRow('p2-000')).toBeInTheDocument();

      // Select a trace on page 2 — now the cross-page selection is {p1-000, p2-000}.
      await user.click(screen.getByRole('checkbox', { name: 'Select trace p2-000' }));

      const actionsButton = await screen.findByRole('button', { name: 'Actions for selected traces' });
      expect(actionsButton).toHaveTextContent('Actions (2)');
      await user.click(actionsButton);
      await user.click(within(await screen.findByRole('menu')).getByRole('menuitem', { name: 'Run scorers' }));

      // The scorer modal launches for the full 2-trace cross-page selection, not the page-2 subset.
      expect(await screen.findByRole('dialog', { name: /Run judge on 2 traces/ })).toBeInTheDocument();
    }, 30000); // heavy: renders two 50-row pages, selects across a page swap, then opens the modal
  });

  describe('trace drawer navigation', () => {
    // The drawer binds ArrowLeft/ArrowRight at the window level; driving it via the keyboard is the
    // user-faithful way to exercise nav (the header's chevron buttons are icon-only, no a11y name).
    test('ArrowRight advances to the next row, writing its V4 long id to the URL', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      await user.click(await findTraceRow('tr-000'));
      await screen.findByRole('dialog');
      expect(new URLSearchParams(lastSearch).get('traceId')).toBe('trace:/cat.sch/tr-000');

      await user.keyboard('{ArrowRight}');
      // Advancing stays a V4 long id (not a bare hex id, which would hit the legacy path).
      await waitFor(() => expect(new URLSearchParams(lastSearch).get('traceId')).toBe('trace:/cat.sch/tr-001'));
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('ArrowLeft on the first row is a no-op (Back disabled at the page start)', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      await user.click(await findTraceRow('tr-000'));
      await screen.findByRole('dialog');

      await user.keyboard('{ArrowLeft}');
      // Still on the first row — there's no previous trace on the page.
      expect(new URLSearchParams(lastSearch).get('traceId')).toBe('trace:/cat.sch/tr-000');
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('ArrowRight on the last row of the page is a no-op (Next disabled at the page end)', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      // The default 3-trace fixture ends at tr-002.
      await user.click(await findTraceRow('tr-002'));
      await screen.findByRole('dialog');

      await user.keyboard('{ArrowRight}');
      expect(new URLSearchParams(lastSearch).get('traceId')).toBe('trace:/cat.sch/tr-002');
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load
  });

  describe('toolbar order', () => {
    // Assert the always-present V4 control ordering. Views pins far left (before the date selector);
    // Detect Issues renders whenever issue detection is enabled (true in OSS) and sits just before
    // Refresh; the selection-only Actions button lands between Columns and Detect Issues.
    test('renders Views → Date → Search → Filter → Columns → Detect Issues → Refresh', async () => {
      renderPage();
      await findTraceRow('tr-000');

      const views = screen.getByTestId('trace-v4-saved-views-trigger');
      const date = screen.getByTestId('time-range-select-dropdown');
      const search = screen.getByPlaceholderText('Search traces by id, input, or output');
      const filter = screen.getByRole('button', { name: /Filters/ });
      const columns = screen.getByRole('button', { name: 'Select visible columns' });
      const detectIssues = screen.getByRole('button', { name: 'Detect issues in traces' });
      const refresh = screen.getByRole('button', { name: 'now' });
      // DOCUMENT_POSITION_FOLLOWING (4) means the arg node comes after `this` node in document order.
      expect(views.compareDocumentPosition(date) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(date.compareDocumentPosition(search) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(search.compareDocumentPosition(filter) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(filter.compareDocumentPosition(columns) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(columns.compareDocumentPosition(detectIssues) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(detectIssues.compareDocumentPosition(refresh) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      // The full page render (incl. the Detect Issues button's first-visit guidance popover) is slow
      // under parallel jsdom load; per-test timeout avoids the lint-forbidden global jest.setTimeout.
    }, 20000);

    // TODO(traces-v4): a "clicking Detect Issues opens the modal" test would drive the shared
    // IssueDetectionModal, which fires gateway endpoint + API-key queries this suite's MSW doesn't
    // stub (it hangs). The button→modal wiring mirrors v3's; presence/order is covered above.

    test('places the selection Actions button between Columns and Refresh', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      await findTraceRow('tr-000');
      await user.click(screen.getByRole('checkbox', { name: 'Select trace tr-000' }));

      const columns = screen.getByRole('button', { name: 'Select visible columns' });
      const actions = await screen.findByRole('button', { name: 'Actions for selected traces' });
      const refresh = screen.getByRole('button', { name: 'now' });
      expect(columns.compareDocumentPosition(actions) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(actions.compareDocumentPosition(refresh) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    }, 20000);
  });
});
