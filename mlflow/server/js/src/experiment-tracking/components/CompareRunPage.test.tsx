import { jest, describe, beforeAll, beforeEach, test, expect } from '@jest/globals';
import { IntlProvider } from 'react-intl';
import { render, screen, waitFor } from '../../common/utils/TestUtils.react18';
import CompareRunPage, { COMPARE_RUNS_SEARCH_RUN_LIMIT } from './CompareRunPage';
import { MockedReduxStoreProvider } from '../../common/utils/TestUtils';
import { setupTestRouter, testRoute, TestRouter } from '../../common/utils/RoutingTestUtils';

import { setupServer } from '../../common/utils/setup-msw';
import { rest } from 'msw';
import { EXPERIMENT_RUNS_MOCK_STORE } from './experiment-page/fixtures/experiment-runs.fixtures';
import { DesignSystemProvider } from '@databricks/design-system';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(60000);

// We're not testing RequestStateWrapper logic so it's just a pass through component in this test
jest.mock('../../common/components/RequestStateWrapper', () => ({
  __esModule: true,
  default: jest.fn(({ children }) => <>{children}</>),
}));

describe('CompareRunPage', () => {
  const { history } = setupTestRouter();
  const apiHandlers = {
    experimentsSuccess: rest.get('/ajax-api/2.0/mlflow/experiments/get', (req, res, ctx) =>
      res(ctx.json({ experiment: {} })),
    ),
    experimentsFailure: rest.get('/ajax-api/2.0/mlflow/experiments/get', (req, res, ctx) =>
      res(ctx.status(404), ctx.json({ message: `Experiment ${req.url.searchParams.get('experiment_id')} not found` })),
    ),
    runsSuccess: rest.get('/ajax-api/2.0/mlflow/runs/get', (req, res, ctx) => {
      return res(ctx.json({ experiments: [] }));
    }),
    runsFailure: rest.get('/ajax-api/2.0/mlflow/runs/get', (req, res, ctx) => {
      return res(ctx.status(404), ctx.json({ message: 'Run was not found' }));
    }),
    searchRunsSuccess: rest.post('/ajax-api/2.0/mlflow/runs/search', (req, res, ctx) => {
      searchRunsRequestBodies.push(req.body as any);
      return res(ctx.json({ runs: [] }));
    }),
    searchRunsFailure: rest.post('/ajax-api/2.0/mlflow/runs/search', (req, res, ctx) => {
      return res(ctx.status(404), ctx.json({ message: 'Run was not found' }));
    }),
    artifactsSuccess: rest.get('/ajax-api/2.0/mlflow/artifacts/list', (req, res, ctx) => {
      return res(ctx.json({}));
    }),
  };

  let searchRunsRequestBodies: { filter?: string; max_results?: number; run_view_type?: string }[] = [];
  let getRunRequestCount = 0;

  const countingRunsHandler = rest.get('/ajax-api/2.0/mlflow/runs/get', (req, res, ctx) => {
    getRunRequestCount += 1;
    return res(ctx.json({ experiments: [] }));
  });

  const server = setupServer(
    // Setup handlers for the API calls
    apiHandlers.artifactsSuccess,
    apiHandlers.experimentsSuccess,
    apiHandlers.searchRunsSuccess,
    apiHandlers.runsSuccess,
  );

  beforeAll(() => {
    server.listen();
  });

  beforeEach(() => {
    searchRunsRequestBodies = [];
    getRunRequestCount = 0;
  });

  const createPageUrl = ({
    experimentIds = ['123456789'],
    runUuids = ['experiment123456789_run1', 'experiment123456789_run2'],
  }: {
    runUuids?: string[];
    experimentIds?: string[];
  } = {}) => {
    const queryParams = new URLSearchParams();
    queryParams.append('runs', JSON.stringify(runUuids));
    queryParams.append('experiments', JSON.stringify(experimentIds));
    return ['/?', queryParams.toString()].join('');
  };

  const renderTestComponent = (routerUrl = createPageUrl()) => {
    render(<CompareRunPage />, {
      wrapper: ({ children }) => (
        <MockedReduxStoreProvider
          state={
            {
              ...EXPERIMENT_RUNS_MOCK_STORE,
              compareExperiments: {},
            } as any
          }
        >
          <IntlProvider locale="en">
            <DesignSystemProvider>
              <TestRouter routes={[testRoute(<>{children}</>, '/')]} history={history} initialEntries={[routerUrl]} />
            </DesignSystemProvider>
          </IntlProvider>
        </MockedReduxStoreProvider>
      ),
    });
  };
  test('should render with minimal props', async () => {
    renderTestComponent();

    await waitFor(() => {
      expect(screen.getByText(/Comparing 2 runs from 1 experiment/i)).toBeInTheDocument();
    });
  });

  test('should render error when experiment is not found', async () => {
    server.resetHandlers(apiHandlers.runsSuccess, apiHandlers.artifactsSuccess, apiHandlers.experimentsFailure);
    renderTestComponent();

    await waitFor(() => {
      expect(screen.getByText(/Experiment 123456789 not found/)).toBeInTheDocument();
    });
  });

  test('should render error when run is not found', async () => {
    server.resetHandlers(apiHandlers.searchRunsFailure, apiHandlers.artifactsSuccess, apiHandlers.experimentsSuccess);
    renderTestComponent();

    await waitFor(() => {
      expect(screen.getByText(/Run was not found/)).toBeInTheDocument();
    });
  });

  test('should fetch all compared runs using a single search request', async () => {
    const runUuids = Array.from({ length: 200 }, (_, index) => `run-${index}`);
    renderTestComponent(createPageUrl({ runUuids }));

    await waitFor(() => {
      expect(searchRunsRequestBodies).toHaveLength(1);
    });

    const [requestBody] = searchRunsRequestBodies;
    expect(requestBody.filter).toBe(`run_id IN (${runUuids.map((runUuid) => `'${runUuid}'`).join(',')})`);
    // Without an explicit max_results the search would be truncated to the default page size
    expect(requestBody.max_results).toBe(runUuids.length);
    // Deleted runs should still show up in the comparison
    expect(requestBody.run_view_type).toBe('ALL');
    expect(getRunRequestCount).toBe(0);
  });

  test('should fall back to fetching runs individually when there are too many to search for', async () => {
    server.resetHandlers(countingRunsHandler, apiHandlers.artifactsSuccess, apiHandlers.experimentsSuccess);
    const runUuids = Array.from({ length: COMPARE_RUNS_SEARCH_RUN_LIMIT + 1 }, (_, index) => `run-${index}`);
    renderTestComponent(createPageUrl({ runUuids }));

    await waitFor(() => {
      expect(getRunRequestCount).toBe(runUuids.length);
    });
    expect(searchRunsRequestBodies).toHaveLength(0);
  });

  test('should display graceful message when URL is malformed', async () => {
    renderTestComponent('?runs=blah&experiments=123');

    await waitFor(() => {
      expect(screen.getByText(/Error while parsing URL(.+)/i)).toBeInTheDocument();
    });
  });
});
