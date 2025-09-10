import { IntlProvider } from 'react-intl';
import { render, screen, waitFor } from '../../common/utils/TestUtils.react18';
import CompareRunPage from './CompareRunPage';
import { MockedReduxStoreProvider } from '../../common/utils/TestUtils';
import { setupTestRouter, testRoute, TestRouter } from '../../common/utils/RoutingTestUtils';
import { setupServer } from '../../common/utils/setup-msw';
import { rest } from 'msw';
import { EXPERIMENT_RUNS_MOCK_STORE } from './experiment-page/fixtures/experiment-runs.fixtures';

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
    artifactsSuccess: rest.get('/ajax-api/2.0/mlflow/artifacts/list', (req, res, ctx) => {
      return res(ctx.json({}));
    }),
  };

  const server = setupServer(
    // Setup handlers for the API calls
    apiHandlers.artifactsSuccess,
    apiHandlers.experimentsSuccess,
    apiHandlers.runsSuccess,
  );

  beforeAll(() => {
    server.listen();
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
            <TestRouter routes={[testRoute(<>{children}</>, '/')]} history={history} initialEntries={[routerUrl]} />
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
    server.resetHandlers(apiHandlers.runsFailure, apiHandlers.artifactsSuccess, apiHandlers.experimentsSuccess);
    renderTestComponent();

    await waitFor(() => {
      expect(screen.getByText(/Run was not found/)).toBeInTheDocument();
    });
  });

  test('should display graceful message when URL is malformed', async () => {
    renderTestComponent('?runs=blah&experiments=123');

    await waitFor(() => {
      expect(screen.getByText(/Error while parsing URL(.+)/i)).toBeInTheDocument();
    });
  });
});
