// @ts-nocheck — punting test typing; see PR2 plan in branch import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { workspaceFetch } from '@databricks/web-shared/spog/workspace-console';
import { queryClientByWorkspace } from '@databricks/web-shared/query-client';
import { setupTestRouter } from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';
import type { Dataset } from '../hooks/useDatasetsQueries';
import { ExperimentEvaluationDatasetsPageV2 } from '../ExperimentEvaluationDatasetsPageV2';
import { renderDatasetsPage } from '../test-utils/renderDatasetsPage';
import { mockEmptyResponse, mockJsonResponse } from '../test-utils/mockResponses';
import { DEFAULT_DATASET_PAGE_SIZE } from '../utils/constants';

// Lightweight stub for the detail route — this file only exercises the list page; the detail
// page has its own test file. Using a stub keeps fetchOrFail (used by useListDatasetRecordsQuery)
// out of the call graph so we only need to mock workspaceFetch.
const DetailRouteStub = () => <div data-testid="datasets-detail-stub">detail</div>;

jest.mock('@databricks/web-shared/spog/workspace-console', () => ({
  workspaceFetch: jest.fn(),
}));

const EXPERIMENT_ID = 'exp-1';
// MLflow routes are mounted under `/ml` inside Databricks (see `createMLflowRoutePath`); the
// page also generates links via `Routes.getExperimentPageDatasetDetailRoute`, which embeds
// that prefix. Tests must use the same prefix so the route registration matches the URL.
const LIST_PATH = '/ml/experiments/:experimentId/datasets';
const DETAIL_PATH = '/ml/experiments/:experimentId/datasets/:datasetId';
const LIST_URL = `/ml/experiments/${EXPERIMENT_ID}/datasets`;
const DETAIL_URL = (datasetId: string) => `/ml/experiments/${EXPERIMENT_ID}/datasets/${datasetId}`;

const datasetOne: Dataset = {
  dataset_id: 'ds-1',
  name: 'Dataset One',
  create_time: '2026-01-01T00:00:00Z',
  last_update_time: '2026-01-02T00:00:00Z',
  created_by: 'alice@databricks.com',
  source_type: 'manual',
};
const datasetTwo: Dataset = {
  dataset_id: 'ds-2',
  name: 'Dataset Two',
  create_time: '2026-01-03T00:00:00Z',
  last_update_time: '2026-01-04T00:00:00Z',
  created_by: 'bob@databricks.com',
  source_type: 'manual',
};

interface MockState {
  datasets: Dataset[];
}

function installWorkspaceFetchMock(state: MockState) {
  jest.mocked(workspaceFetch).mockImplementation(async (input, init) => {
    const urlStr = input.toString();
    const method = init?.method ?? 'GET';

    // DELETE /datasets/{id} — strip the deleted row from the in-memory store.
    const deleteMatch = urlStr.match(/\/managed-evals\/datasets\/([^/?]+)(?:\?|$)/);
    if (deleteMatch && method === 'DELETE') {
      state.datasets = state.datasets.filter((d) => d.dataset_id !== deleteMatch[1]);
      return mockEmptyResponse();
    }

    // GET /datasets/{id} — used by the detail-page query when navigating.
    if (deleteMatch && method === 'GET') {
      const found = state.datasets.find((d) => d.dataset_id === deleteMatch[1]);
      if (!found) {
        return mockJsonResponse(
          { error_code: 'NOT_FOUND', message: 'Not found' },
          { status: 404, statusText: 'Not Found' },
        );
      }
      return mockJsonResponse(found);
    }

    // GET /datasets?filter=... — V2 paginated list. Slices `state.datasets` by `page_token`
    // and `page_size` so cursor-pagination tests can step through real-feeling pages without
    // any extra setup; tests with ≤ page_size datasets continue to see a single-page response.
    if (urlStr.includes('/managed-evals/datasets?') && method === 'GET') {
      // Don't construct `new URL(urlStr)` — the list query passes a relative path through
      // `getAjaxUrl`, which throws "Invalid URL" without a base. Parse the query string only.
      const queryStart = urlStr.indexOf('?');
      const queryParams = queryStart >= 0 ? new URLSearchParams(urlStr.slice(queryStart + 1)) : new URLSearchParams();
      const pageSize = Number(queryParams.get('page_size')) || DEFAULT_DATASET_PAGE_SIZE;
      const startIdx = Number(queryParams.get('page_token')) || 0;
      const slice = state.datasets.slice(startIdx, startIdx + pageSize);
      const nextStartIdx = startIdx + pageSize;
      return mockJsonResponse({
        datasets: slice,
        next_page_token: nextStartIdx < state.datasets.length ? String(nextStartIdx) : undefined,
      });
    }

    return mockEmptyResponse();
  });
}

const datasetsRoutes = [
  { path: LIST_PATH, element: <ExperimentEvaluationDatasetsPageV2 /> },
  { path: DETAIL_PATH, element: <DetailRouteStub /> },
];

describe('DatasetsListPage', () => {
  // setupTestRouter registers beforeAll/afterAll/beforeEach hooks, so it MUST live at describe
  // scope — calling it inside a test throws "Hooks cannot be defined inside tests".
  const { history } = setupTestRouter();
  let state: MockState;

  beforeEach(() => {
    state = { datasets: [datasetOne, datasetTwo] };
    installWorkspaceFetchMock(state);
    // The page's MetastoreClientProvider routes useDatasetsPageQuery through the shared
    // workspace-scoped QueryClient singleton (not the test's own QueryClient). With
    // `staleTime: Infinity`, cached responses from a prior test would otherwise satisfy the
    // next mount without refetching against the freshly-installed mock — leading to flake
    // where, e.g., test #2 sees the empty state cached by test #1.
    queryClientByWorkspace.getActiveValue().clear();
  });

  afterEach(() => {
    // Guard against tests that bail out before their own `useRealTimers()` runs — leaked fake
    // timers freeze React Query's setTimeout-based refetches, leaving the next test stuck in
    // a loading state.
    jest.useRealTimers();
    jest.clearAllMocks();
  });

  test('renders only the CTA (no toolbar) when the workspace has no datasets', async () => {
    state.datasets = [];
    renderDatasetsPage({
      initialUrl: LIST_URL,
      routes: datasetsRoutes,
      history,
    });

    expect(await screen.findByRole('heading', { name: /Create an evaluation dataset/i })).toBeInTheDocument();
    // The CTA owns the only Create-dataset button; the toolbar would otherwise duplicate it.
    expect(screen.getAllByRole('button', { name: /Create dataset/i })).toHaveLength(1);
    // Toolbar (search + refresh) must not render in the bare-CTA state — those affordances
    // only become useful once at least one dataset exists or a search is active.
    expect(screen.queryByPlaceholderText(/Search datasets/i)).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Refresh datasets/i })).not.toBeInTheDocument();
  });

  test('renders rows for each dataset and exposes per-row links to the detail page', async () => {
    renderDatasetsPage({
      initialUrl: LIST_URL,
      routes: datasetsRoutes,
      history,
    });

    expect(await screen.findByRole('link', { name: 'Dataset One' })).toHaveAttribute(
      'href',
      DETAIL_URL(datasetOne.dataset_id),
    );
    expect(screen.getByRole('link', { name: 'Dataset Two' })).toBeInTheDocument();
  });

  test('clicking a row link navigates to the detail page', async () => {
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: LIST_URL,
      routes: datasetsRoutes,
      history,
    });

    const link = await screen.findByRole('link', { name: 'Dataset One' });
    await user.click(link);

    await waitFor(() => {
      expect(history.location.pathname).toBe(DETAIL_URL(datasetOne.dataset_id));
    });
  });

  test('pressing Enter on the search input writes `q` to the URL and refetches', async () => {
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: LIST_URL,
      routes: datasetsRoutes,
      history,
    });

    await screen.findByRole('link', { name: 'Dataset One' });

    const searchInput = screen.getByPlaceholderText(/Search datasets/i);
    // Typing alone is a local update; the URL must not change until Enter is pressed.
    await user.type(searchInput, 'One');
    expect(history.location.search).not.toContain('q=');

    // AntD's `onPressEnter` is gated on `keyCode === 13` (its legacy keydown handler).
    // userEvent's keyboard synthesis sets `key`/`code` but not the legacy keyCode in all jsdom
    // versions — so fire keyDown directly with the keyCode to exercise AntD's path reliably.
    fireEvent.keyDown(searchInput, { key: 'Enter', code: 'Enter', keyCode: 13 });

    await waitFor(() => {
      expect(history.location.search).toContain('q=One');
    });
  });

  test('typing into the search input without pressing Enter does NOT write `q` to the URL', async () => {
    // The endpoint is rate-limited, so we deliberately avoid an as-you-type/debounce model;
    // input value lives in component state until the user submits explicitly. This test
    // pins that contract so we don't regress to a debounced/auto-search behavior.
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: LIST_URL,
      routes: datasetsRoutes,
      history,
    });

    const link = await screen.findByRole('link', { name: 'Dataset One' });
    const searchInput = screen.getByPlaceholderText(/Search datasets/i);

    await user.type(searchInput, 'abc');
    // Navigating away (or just letting the page sit) must not surface `q` in the URL.
    await user.click(link);

    await waitFor(() => {
      expect(history.location.pathname).toBe(DETAIL_URL(datasetOne.dataset_id));
    });
    expect(history.location.search).not.toContain('q=');
  });

  test('pressing Enter swaps the table body for a skeleton until the new results arrive', async () => {
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: LIST_URL,
      routes: datasetsRoutes,
      history,
    });

    await screen.findByRole('link', { name: 'Dataset One' });

    // Hold the next fetch open so we can assert the in-flight skeleton state. The fallback
    // implementation (set up in beforeEach) handles every other call.
    let resolveSearchFetch!: (value: Response) => void;
    jest.mocked(workspaceFetch).mockImplementationOnce(
      () =>
        new Promise<Response>((resolve) => {
          resolveSearchFetch = resolve;
        }),
    );

    const searchInput = screen.getByPlaceholderText(/Search datasets/i);
    await user.type(searchInput, 'Two');
    // AntD's onPressEnter checks the legacy `keyCode` — fire keyDown directly so the test
    // doesn't depend on userEvent setting that field.
    fireEvent.keyDown(searchInput, { key: 'Enter', code: 'Enter', keyCode: 13 });

    // Previous data rows are replaced by skeletons; the table region reports aria-busy.
    await waitFor(() => {
      expect(screen.queryByRole('link', { name: 'Dataset One' })).not.toBeInTheDocument();
    });
    expect(screen.getByRole('region', { name: /Datasets/i })).toHaveAttribute('aria-busy', 'true');
    // Header row is preserved.
    expect(screen.getByRole('columnheader', { name: /^Name$/i })).toBeInTheDocument();

    // Resolve the search and assert the new data renders.
    resolveSearchFetch(mockJsonResponse({ datasets: [datasetTwo], next_page_token: undefined }));
    expect(await screen.findByRole('link', { name: 'Dataset Two' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: /Datasets/i })).toHaveAttribute('aria-busy', 'false');
  });

  test('clicking the clear button on a non-empty search shows the skeleton until results arrive', async () => {
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: `${LIST_URL}?q=One`,
      routes: datasetsRoutes,
      history,
    });

    await screen.findByRole('link', { name: 'Dataset One' });

    // Hold the post-clear fetch open to assert the skeleton state.
    let resolveClearFetch!: (value: Response) => void;
    jest.mocked(workspaceFetch).mockImplementationOnce(
      () =>
        new Promise<Response>((resolve) => {
          resolveClearFetch = resolve;
        }),
    );

    // Du Bois Input renders the X affordance with the accessible name `close-circle`
    // (see design-system/Input/Input.test.tsx — "calls onClear when clear button is clicked").
    await user.click(screen.getByLabelText('close-circle'));

    await waitFor(() => {
      expect(screen.queryByRole('link', { name: 'Dataset One' })).not.toBeInTheDocument();
    });
    expect(screen.getByRole('region', { name: /Datasets/i })).toHaveAttribute('aria-busy', 'true');

    resolveClearFetch(mockJsonResponse({ datasets: [datasetOne, datasetTwo], next_page_token: undefined }));
    await screen.findByRole('link', { name: 'Dataset One' });
    expect(screen.getByRole('region', { name: /Datasets/i })).toHaveAttribute('aria-busy', 'false');
  });

  test('refresh button refetches in the background — prior rows stay visible, no skeleton', async () => {
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: LIST_URL,
      routes: datasetsRoutes,
      history,
    });

    await screen.findByRole('link', { name: 'Dataset One' });

    // Hold the refresh fetch open: refresh is a refetch with an unchanged search filter, so
    // the in-flight UX must keep the prior rows on screen and never flip to the skeleton.
    let resolveRefresh!: (value: Response) => void;
    jest.mocked(workspaceFetch).mockImplementationOnce(
      () =>
        new Promise<Response>((resolve) => {
          resolveRefresh = resolve;
        }),
    );

    await user.click(screen.getByRole('button', { name: /Refresh datasets/i }));

    // Give React a tick to start the refetch; prior rows must remain visible.
    await waitFor(() => {
      expect(screen.getByRole('link', { name: 'Dataset One' })).toBeInTheDocument();
    });
    expect(screen.getByRole('region', { name: /Datasets/i })).toHaveAttribute('aria-busy', 'false');

    resolveRefresh(mockJsonResponse({ datasets: [datasetOne, datasetTwo], next_page_token: undefined }));
  });

  test('refresh button re-fetches the datasets list', async () => {
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: LIST_URL,
      routes: datasetsRoutes,
      history,
    });

    await screen.findByRole('link', { name: 'Dataset One' });
    const callsBeforeRefresh = jest.mocked(workspaceFetch).mock.calls.length;

    await user.click(screen.getByRole('button', { name: /Refresh datasets/i }));

    await waitFor(() => {
      expect(jest.mocked(workspaceFetch).mock.calls.length).toBeGreaterThan(callsBeforeRefresh);
    });
  });

  describe('cursor pagination', () => {
    // Seeded dataset count > DEFAULT_DATASET_PAGE_SIZE so the response advertises a
    // `next_page_token` and the CursorPagination footer renders. Page 1 fills, page 2 is partial.
    const TOTAL = DEFAULT_DATASET_PAGE_SIZE + 5;
    const padId = (i: number) => String(i).padStart(2, '0');
    const FIRST_ON_PAGE_1 = padId(0);
    const FIRST_ON_PAGE_2 = padId(DEFAULT_DATASET_PAGE_SIZE);
    const LAST_ON_PAGE_2 = padId(TOTAL - 1);

    const buildPaginationDatasets = (count: number): Dataset[] =>
      Array.from({ length: count }, (_, i) => ({
        dataset_id: `ds-${padId(i)}`,
        name: `Dataset ${padId(i)}`,
        create_time: '2026-01-01T00:00:00Z',
        last_update_time: '2026-01-01T00:00:00Z',
        created_by: 'alice@databricks.com',
        source_type: 'manual',
      }));

    test('clicking Next advances to the next page and reveals page-2 datasets', async () => {
      state.datasets = buildPaginationDatasets(TOTAL);
      const user = userEvent.setup();
      renderDatasetsPage({
        initialUrl: LIST_URL,
        routes: datasetsRoutes,
        history,
      });

      // Page 1: first dataset visible; first page-2 dataset hidden.
      expect(await screen.findByRole('link', { name: `Dataset ${FIRST_ON_PAGE_1}` })).toBeInTheDocument();
      expect(screen.queryByRole('link', { name: `Dataset ${FIRST_ON_PAGE_2}` })).not.toBeInTheDocument();

      // Du Bois CursorPagination button accessible name is "Next" (see
      // design-system/Pagination/CursorPagination.test.tsx).
      await user.click(screen.getByRole('button', { name: 'Next' }));

      // Page 2 datasets appear; the page-1 head is no longer rendered.
      expect(await screen.findByRole('link', { name: `Dataset ${FIRST_ON_PAGE_2}` })).toBeInTheDocument();
      expect(screen.getByRole('link', { name: `Dataset ${LAST_ON_PAGE_2}` })).toBeInTheDocument();
      expect(screen.queryByRole('link', { name: `Dataset ${FIRST_ON_PAGE_1}` })).not.toBeInTheDocument();
    });

    test('while the next-page fetch is in flight, rows swap for skeletons and both pagination buttons disable', async () => {
      state.datasets = buildPaginationDatasets(TOTAL);
      const user = userEvent.setup();
      renderDatasetsPage({
        initialUrl: LIST_URL,
        routes: datasetsRoutes,
        history,
      });

      await screen.findByRole('link', { name: `Dataset ${FIRST_ON_PAGE_1}` });

      // Hold the next-page fetch open so we can assert the in-flight skeleton state. The
      // fallback installWorkspaceFetchMock handler still serves any other calls.
      let resolveNextPageFetch!: (value: Response) => void;
      jest.mocked(workspaceFetch).mockImplementationOnce(
        () =>
          new Promise<Response>((resolve) => {
            resolveNextPageFetch = resolve;
          }),
      );

      await user.click(screen.getByRole('button', { name: 'Next' }));

      // Page-1 rows replaced by skeletons; the table region reports aria-busy; both
      // pagination buttons disable so the user can't queue up clicks against stale data.
      await waitFor(() => {
        expect(screen.queryByRole('link', { name: `Dataset ${FIRST_ON_PAGE_1}` })).not.toBeInTheDocument();
      });
      expect(screen.getByRole('region', { name: /Datasets/i })).toHaveAttribute('aria-busy', 'true');
      expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled();
      expect(screen.getByRole('button', { name: 'Previous' })).toBeDisabled();

      // Resolve the page-2 fetch and assert the table returns to a normal, settled state.
      resolveNextPageFetch(
        mockJsonResponse({
          datasets: state.datasets.slice(DEFAULT_DATASET_PAGE_SIZE, TOTAL),
          next_page_token: undefined,
        }),
      );
      expect(await screen.findByRole('link', { name: `Dataset ${FIRST_ON_PAGE_2}` })).toBeInTheDocument();
      expect(screen.getByRole('region', { name: /Datasets/i })).toHaveAttribute('aria-busy', 'false');
      // Previous re-enables (we're on page 2). Next stays disabled because page 2 is the
      // last page and the server's response carried no `next_page_token`.
      expect(screen.getByRole('button', { name: 'Previous' })).not.toBeDisabled();
    });

    test('clicking Previous from page 2 returns to page 1 and re-renders the original slice', async () => {
      state.datasets = buildPaginationDatasets(TOTAL);
      const user = userEvent.setup();
      renderDatasetsPage({
        initialUrl: LIST_URL,
        routes: datasetsRoutes,
        history,
      });

      await screen.findByRole('link', { name: `Dataset ${FIRST_ON_PAGE_1}` });
      await user.click(screen.getByRole('button', { name: 'Next' }));
      await screen.findByRole('link', { name: `Dataset ${FIRST_ON_PAGE_2}` });

      await user.click(screen.getByRole('button', { name: 'Previous' }));

      // Original page-1 head is back; page-2 head is gone.
      expect(await screen.findByRole('link', { name: `Dataset ${FIRST_ON_PAGE_1}` })).toBeInTheDocument();
      expect(screen.queryByRole('link', { name: `Dataset ${FIRST_ON_PAGE_2}` })).not.toBeInTheDocument();
    });
  });

  test('row dropdown → Delete → confirm modal removes the row and shows a success toast', async () => {
    const user = userEvent.setup();
    renderDatasetsPage({
      initialUrl: LIST_URL,
      routes: datasetsRoutes,
      history,
    });

    // The per-row action is a dropdown trigger (matches v1's overflow-menu pattern); open it
    // to reveal the "Delete" item.
    const actionsTrigger = await screen.findByRole('button', { name: `Actions for dataset ${datasetOne.name}` });
    await user.click(actionsTrigger);
    const deleteItem = await screen.findByRole('menuitem', { name: /Delete/i });
    await user.click(deleteItem);

    // The DangerModal renders an OK button labeled "Delete".
    const confirmButton = await screen.findByRole('button', { name: /^Delete$/ });
    await user.click(confirmButton);

    await waitFor(() => {
      expect(screen.queryByRole('link', { name: 'Dataset One' })).not.toBeInTheDocument();
    });
    expect(screen.getByText(/Deleted dataset "Dataset One"/i)).toBeInTheDocument();
    // Surviving row still rendered.
    expect(screen.getByRole('link', { name: 'Dataset Two' })).toBeInTheDocument();
  });
});