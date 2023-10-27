import { act, renderWithIntl, screen } from '../../common/utils/TestUtils';
import { RunView } from './RunView';
import thunk from 'redux-thunk';
import configureStore from 'redux-mock-store';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { EXPERIMENT_RUNS_MOCK_STORE } from './experiment-page/fixtures/experiment-runs.fixtures';
import { MemoryRouter, Route, Routes } from '../../common/utils/RoutingUtils';

import { shouldEnableDeepLearningUI } from '../../common/utils/FeatureUtils';
import userEvent from '@testing-library/user-event';
import { RoutePaths } from '../routes';

// Mock tab contents
jest.mock('./run-page/RunViewMetricCharts', () => ({
  RunViewMetricCharts: jest.fn(() => <div>metric charts tab</div>),
}));
jest.mock('./run-page/RunViewOverview', () => ({
  RunViewOverview: jest.fn(() => <div>overview tab</div>),
}));
jest.mock('./run-page/RunViewArtifactTab', () => ({
  RunViewArtifactTab: jest.fn(() => <div>artifacts tab</div>),
}));

// Enable flag manipulation
jest.mock('../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual('../../common/utils/FeatureUtils'),
  shouldEnableDeepLearningUI: jest.fn(),
}));

describe('RunView navigation integration test', () => {
  const minimalProps = {
    runUuid: 'experiment123456789_run1',
    getMetricPagePath: jest.fn().mockReturnValue('/'),
    experimentId: '123456789',
    handleSetRunTag: jest.fn(),
  };
  const renderComponent = (
    initialRoute = '/experiments/123456789/runs/experiment123456789_run1',
  ) => {
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
                <RunView {...minimalProps} />
              </Provider>
            }
          />
        </Routes>
      </MemoryRouter>,
    );
  };
  beforeEach(() => {
    jest.mocked(shouldEnableDeepLearningUI).mockImplementation(() => true);
  });
  test('should display overview by default and allow changing the tab', async () => {
    renderComponent();
    expect(screen.queryByText('overview tab')).toBeInTheDocument();
    expect(screen.queryByText('metric charts tab')).not.toBeInTheDocument();
    expect(screen.queryByText('artifacts tab')).not.toBeInTheDocument();

    await act(async () => {
      userEvent.click(screen.getByRole('tab', { name: 'Metric charts' }));
    });

    expect(screen.queryByText('overview tab')).not.toBeInTheDocument();
    expect(screen.queryByText('metric charts tab')).toBeInTheDocument();
    expect(screen.queryByText('artifacts tab')).not.toBeInTheDocument();

    await act(async () => {
      userEvent.click(screen.getByRole('tab', { name: 'Artifacts' }));
    });

    expect(screen.queryByText('overview tab')).not.toBeInTheDocument();
    expect(screen.queryByText('metric charts tab')).not.toBeInTheDocument();
    expect(screen.queryByText('artifacts tab')).toBeInTheDocument();
  });

  test('should not display navigation tabs when deep learning UI features are not enabled', async () => {
    jest.mocked(shouldEnableDeepLearningUI).mockImplementation(() => false);

    renderComponent();

    expect(screen.queryByRole('tab', { name: 'Overview' })).not.toBeInTheDocument();
    expect(screen.queryByRole('tab', { name: 'Metric charts' })).not.toBeInTheDocument();
    expect(screen.queryByRole('tab', { name: 'Artifacts' })).not.toBeInTheDocument();

    expect(screen.queryByText('overview tab')).toBeInTheDocument();
  });

  test('should display artirfact tab if using a targeted artifact URL', async () => {
    renderComponent(
      '/experiments/123456789/runs/experiment123456789_run1/artifacts/model/conda.yaml',
    );
    expect(screen.queryByText('artifacts tab')).toBeInTheDocument();
  });

  test('should display artirfact tab if using a targeted artifact URL (legacy artifactPath pattern)', async () => {
    renderComponent(
      '/experiments/123456789/runs/experiment123456789_run1/artifactPath/model/conda.yaml',
    );
    expect(screen.queryByText('artifacts tab')).toBeInTheDocument();
  });
});
