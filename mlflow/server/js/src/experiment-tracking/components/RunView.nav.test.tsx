import { renderWithIntl, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { RunPage } from './run-page/RunPage';
import thunk from 'redux-thunk';
import configureStore from 'redux-mock-store';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { EXPERIMENT_RUNS_MOCK_STORE } from './experiment-page/fixtures/experiment-runs.fixtures';
import { createMLflowRoutePath } from '../../common/utils/RoutingUtils';
import { testRoute, TestRouter } from '../../common/utils/RoutingTestUtils';
import userEvent from '@testing-library/user-event';
import { RoutePaths } from '../routes';
import { useRunDetailsPageData } from './run-page/hooks/useRunDetailsPageData';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

// Mock tab contents
jest.mock('./run-page/RunViewMetricCharts', () => ({
  RunViewMetricCharts: jest.fn((props) => <div>{props.mode} metric charts</div>),
}));
jest.mock('./run-page/RunViewOverview', () => ({
  RunViewOverview: jest.fn(() => <div>overview tab</div>),
}));
jest.mock('./run-page/RunViewArtifactTab', () => ({
  RunViewArtifactTab: jest.fn(() => <div>artifacts tab</div>),
}));
jest.mock('./run-page/RunViewHeaderRegisterModelButton', () => ({
  RunViewHeaderRegisterModelButton: jest.fn(() => <div>register model</div>),
}));
jest.mock('./run-page/hooks/useRunDetailsPageData', () => ({
  useRunDetailsPageData: jest.fn(),
}));

describe('RunView navigation integration test', () => {
  const renderComponent = (initialRoute = '/experiments/123456789/runs/experiment123456789_run1') => {
    const mockStore = configureStore([thunk, promiseMiddleware()]);
    const mockState = {
      ...EXPERIMENT_RUNS_MOCK_STORE,
      compareExperiments: {
        comparedExperimentIds: [],
        hasComparedExperimentsBefore: false,
      },
    };
    const queryClient = new QueryClient();
    const renderResult = renderWithIntl(
      <Provider store={mockStore(mockState)}>
        <QueryClientProvider client={queryClient}>
          <TestRouter
            initialEntries={[createMLflowRoutePath(initialRoute)]}
            routes={[testRoute(<RunPage />, RoutePaths.runPageWithTab)]}
          />
        </QueryClientProvider>
      </Provider>,
    );

    return renderResult;
  };
  beforeEach(() => {
    jest.mocked(useRunDetailsPageData).mockImplementation(
      () =>
        ({
          experiment: EXPERIMENT_RUNS_MOCK_STORE.entities.experimentsById['123456789'],
          runInfo: EXPERIMENT_RUNS_MOCK_STORE.entities.runInfosByUuid['experiment123456789_run1'],
          latestMetrics: {},
          tags: {},
          params: {},
          error: null,
          loading: false,
        } as any),
    );
  });
  test('should display overview by default and allow changing the tab', async () => {
    renderComponent();

    await waitFor(() => {
      expect(screen.queryByText('overview tab')).toBeInTheDocument();
      expect(screen.queryByText('model metric charts')).not.toBeInTheDocument();
      expect(screen.queryByText('system metric charts')).not.toBeInTheDocument();
      expect(screen.queryByText('artifacts tab')).not.toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole('tab', { name: 'Model metrics' }));

    expect(screen.queryByText('overview tab')).not.toBeInTheDocument();
    expect(screen.queryByText('model metric charts')).toBeInTheDocument();
    expect(screen.queryByText('system metric charts')).not.toBeInTheDocument();
    expect(screen.queryByText('artifacts tab')).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('tab', { name: 'System metrics' }));

    expect(screen.queryByText('overview tab')).not.toBeInTheDocument();
    expect(screen.queryByText('model metric charts')).not.toBeInTheDocument();
    expect(screen.queryByText('system metric charts')).toBeInTheDocument();
    expect(screen.queryByText('artifacts tab')).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('tab', { name: 'Artifacts' }));

    expect(screen.queryByText('overview tab')).not.toBeInTheDocument();
    expect(screen.queryByText('model metrics')).not.toBeInTheDocument();
    expect(screen.queryByText('system metrics')).not.toBeInTheDocument();
    expect(screen.queryByText('artifacts tab')).toBeInTheDocument();
  });

  test('should display artirfact tab if using a targeted artifact URL', async () => {
    renderComponent('/experiments/123456789/runs/experiment123456789_run1/artifacts/model/conda.yaml');

    await waitFor(() => {
      expect(screen.queryByText('artifacts tab')).toBeInTheDocument();
    });
  });

  test('should display artirfact tab if using a targeted artifact URL (legacy artifactPath pattern)', async () => {
    renderComponent('/experiments/123456789/runs/experiment123456789_run1/artifactPath/model/conda.yaml');
    await waitFor(() => {
      expect(screen.queryByText('artifacts tab')).toBeInTheDocument();
    });
  });
});
