import { renderWithIntl, act, screen } from 'common/utils/TestUtils.react17';
import { RunPageV2 } from './run-page/RunPageV2';
import thunk from 'redux-thunk';
import configureStore from 'redux-mock-store';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { EXPERIMENT_RUNS_MOCK_STORE } from './experiment-page/fixtures/experiment-runs.fixtures';
import { MemoryRouter, Route, Routes } from '../../common/utils/RoutingUtils';

import { shouldEnableDeepLearningUI } from '../../common/utils/FeatureUtils';
import userEvent from '@testing-library/user-event';
import { RoutePaths } from '../routes';
import { useRunDetailsPageData } from './run-page/useRunDetailsPageData';

// Mock tab contents
jest.mock('./run-page/RunViewMetricCharts', () => ({
  RunViewMetricCharts: jest.fn((props) => <div>{props.mode} metric charts</div>),
}));
jest.mock('./run-page/RunViewOverview', () => ({
  RunViewOverview: jest.fn(() => <div>overview tab</div>),
}));
jest.mock('./run-page/RunViewOverviewV2', () => ({
  RunViewOverviewV2: jest.fn(() => <div>overview tab</div>),
}));
jest.mock('./run-page/RunViewArtifactTab', () => ({
  RunViewArtifactTab: jest.fn(() => <div>artifacts tab</div>),
}));
jest.mock('./run-page/RunViewHeaderRegisterModelButton', () => ({
  RunViewHeaderRegisterModelButton: jest.fn(() => <div>register model</div>),
}));
jest.mock('./run-page/useRunDetailsPageData', () => ({
  useRunDetailsPageData: jest.fn(),
}));

// Enable flag manipulation
jest.mock('../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual('../../common/utils/FeatureUtils'),
  shouldEnableDeepLearningUI: jest.fn(),
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
    return renderWithIntl(
      <MemoryRouter initialEntries={[initialRoute]}>
        <Routes>
          <Route
            path={RoutePaths.runPageWithTab}
            element={
              <Provider store={mockStore(mockState)}>
                <RunPageV2 />
              </Provider>
            }
          />
        </Routes>
      </MemoryRouter>,
    );
  };
  beforeEach(() => {
    jest.mocked(shouldEnableDeepLearningUI).mockImplementation(() => true);
    jest.mocked(useRunDetailsPageData).mockImplementation(
      () =>
        ({
          data: {
            experiment: EXPERIMENT_RUNS_MOCK_STORE.entities.experimentsById['123456789'],
            runInfo: EXPERIMENT_RUNS_MOCK_STORE.entities.runInfosByUuid['experiment123456789_run1'],
            latestMetrics: {},
            tags: {},
          },
          errors: {},
          loading: false,
        } as any),
    );
  });
  test('should display overview by default and allow changing the tab', async () => {
    renderComponent();
    expect(screen.queryByText('overview tab')).toBeInTheDocument();
    expect(screen.queryByText('model metric charts')).not.toBeInTheDocument();
    expect(screen.queryByText('system metric charts')).not.toBeInTheDocument();
    expect(screen.queryByText('artifacts tab')).not.toBeInTheDocument();

    await act(async () => {
      userEvent.click(screen.getByRole('tab', { name: 'Model metrics' }));
    });

    expect(screen.queryByText('overview tab')).not.toBeInTheDocument();
    expect(screen.queryByText('model metric charts')).toBeInTheDocument();
    expect(screen.queryByText('system metric charts')).not.toBeInTheDocument();
    expect(screen.queryByText('artifacts tab')).not.toBeInTheDocument();

    await act(async () => {
      userEvent.click(screen.getByRole('tab', { name: 'System metrics' }));
    });

    expect(screen.queryByText('overview tab')).not.toBeInTheDocument();
    expect(screen.queryByText('model metric charts')).not.toBeInTheDocument();
    expect(screen.queryByText('system metric charts')).toBeInTheDocument();
    expect(screen.queryByText('artifacts tab')).not.toBeInTheDocument();

    await act(async () => {
      userEvent.click(screen.getByRole('tab', { name: 'Artifacts' }));
    });

    expect(screen.queryByText('overview tab')).not.toBeInTheDocument();
    expect(screen.queryByText('model metrics')).not.toBeInTheDocument();
    expect(screen.queryByText('system metrics')).not.toBeInTheDocument();
    expect(screen.queryByText('artifacts tab')).toBeInTheDocument();
  });

  test('should display artirfact tab if using a targeted artifact URL', async () => {
    renderComponent('/experiments/123456789/runs/experiment123456789_run1/artifacts/model/conda.yaml');
    expect(screen.queryByText('artifacts tab')).toBeInTheDocument();
  });

  test('should display artirfact tab if using a targeted artifact URL (legacy artifactPath pattern)', async () => {
    renderComponent('/experiments/123456789/runs/experiment123456789_run1/artifactPath/model/conda.yaml');
    expect(screen.queryByText('artifacts tab')).toBeInTheDocument();
  });
});
