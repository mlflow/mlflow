import { jest, describe, test, expect } from '@jest/globals';
import { render, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { IntlProvider } from 'react-intl';
import { Provider } from 'react-redux';
import { applyMiddleware, compose, createStore } from 'redux';
import promiseMiddleware from 'redux-promise-middleware';
import { DesignSystemProvider } from '@databricks/design-system';

import { MemoryRouter } from '../../../../../common/utils/RoutingUtils';
import { EXPERIMENT_RUNS_MOCK_STORE } from '../../fixtures/experiment-runs.fixtures';
import { ExperimentViewRunsControls } from './ExperimentViewRunsControls';
import { experimentRunsSelector } from '../../utils/experimentRuns.selector';
import { createExperimentPageUIState } from '../../models/ExperimentPageUIState';
import { createExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';

// The saved-views dropdown is exercised in its own suite; here we only assert the runs toolbar
// mounts (or gates) it, so stub it to a cheap sentinel that doesn't need redux tag state.
jest.mock('../header/ExperimentViewSavedViewsButton', () => ({
  ExperimentViewSavedViewsButton: () => <div data-testid="saved-views-trigger" />,
}));

const MOCK_EXPERIMENT_ID = '123456789';
const MOCK_EXPERIMENT = EXPERIMENT_RUNS_MOCK_STORE.entities.experimentsById[MOCK_EXPERIMENT_ID];
const MOCK_RUNS_DATA = experimentRunsSelector(EXPERIMENT_RUNS_MOCK_STORE, { experiments: [MOCK_EXPERIMENT] });

const renderControls = (props: Partial<React.ComponentProps<typeof ExperimentViewRunsControls>> = {}) =>
  render(
    <Provider
      store={createStore((s) => s as any, EXPERIMENT_RUNS_MOCK_STORE, compose(applyMiddleware(promiseMiddleware())))}
    >
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <MemoryRouter>
            <ExperimentViewRunsControls
              runsData={MOCK_RUNS_DATA}
              searchFacetsState={createExperimentPageSearchFacetsState()}
              experimentId={MOCK_EXPERIMENT_ID}
              viewState={{
                runsSelected: {},
                columnSelectorVisible: false,
                hiddenChildRunsSelected: {},
                previewPaneVisible: false,
                artifactViewState: {},
              }}
              updateViewState={() => {}}
              requestError={null}
              expandRows={false}
              updateExpandRows={() => {}}
              refreshRuns={() => {}}
              uiState={createExperimentPageUIState()}
              isLoading={false}
              isComparingExperiments={false}
              onResetColumns={() => {}}
              {...props}
            />
          </MemoryRouter>
        </DesignSystemProvider>
      </IntlProvider>
    </Provider>,
  );

describe('ExperimentViewRunsControls saved-views placement', () => {
  test('renders the saved-views button in the runs toolbar', () => {
    renderControls();
    expect(screen.getByTestId('saved-views-trigger')).toBeInTheDocument();
  });

  test('hides the saved-views button when comparing multiple experiments', () => {
    renderControls({ isComparingExperiments: true });
    expect(screen.queryByTestId('saved-views-trigger')).not.toBeInTheDocument();
  });

  test('hides the saved-views button when the experiment is absent from the store', () => {
    renderControls({ experimentId: 'not-in-store' });
    expect(screen.queryByTestId('saved-views-trigger')).not.toBeInTheDocument();
  });
});
