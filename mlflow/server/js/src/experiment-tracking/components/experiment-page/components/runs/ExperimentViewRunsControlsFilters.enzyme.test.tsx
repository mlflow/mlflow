import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { BrowserRouter } from '../../../../../common/utils/RoutingUtils';
import { mountWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';
import { EXPERIMENT_RUNS_MOCK_STORE } from '../../fixtures/experiment-runs.fixtures';
import { experimentRunsSelector } from '../../utils/experimentRuns.selector';
import type { ExperimentViewRunsControlsFiltersProps } from './ExperimentViewRunsControlsFilters';
import { ExperimentViewRunsControlsFilters } from './ExperimentViewRunsControlsFilters';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { createExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';

jest.mock('../../../evaluation-artifacts-compare/EvaluationCreatePromptRunModal', () => ({
  EvaluationCreatePromptRunModal: () => <div />,
}));

const MOCK_EXPERIMENT = EXPERIMENT_RUNS_MOCK_STORE.entities.experimentsById['123456789'];

const MOCK_RUNS_DATA = experimentRunsSelector(EXPERIMENT_RUNS_MOCK_STORE, {
  experiments: [MOCK_EXPERIMENT],
});

jest.mock('./ExperimentViewRunsColumnSelector', () => ({
  ExperimentViewRunsColumnSelector: () => <div />,
}));

const mockStore = configureStore([thunk, promiseMiddleware()]);
const minimalStore = mockStore({
  entities: {
    datasetsByExperimentId: {},
    experimentsById: {},
  },
  apis: jest.fn((key) => {
    return {};
  }),
});

const doSimpleMock = (props: ExperimentViewRunsControlsFiltersProps) =>
  mountWithIntl(
    <Provider store={minimalStore}>
      <DesignSystemProvider>
        <IntlProvider locale="en">
          <BrowserRouter>
            <ExperimentViewRunsControlsFilters {...props} />
          </BrowserRouter>
        </IntlProvider>
      </DesignSystemProvider>
    </Provider>,
  );

describe('ExperimentViewRunsControlsFilters', () => {
  test('should render with given search facets model properly', () => {
    const searchFacetsState = createExperimentPageSearchFacetsState();

    const wrapper = doSimpleMock({
      runsData: MOCK_RUNS_DATA,
      experimentId: '123456789',
      onDownloadCsv: () => {},
      viewState: { runsSelected: {}, viewMaximized: false } as any,
      updateViewState: () => {},
      searchFacetsState,
      requestError: null,
      refreshRuns: () => {},
      viewMaximized: false,
    });

    expect(wrapper).toBeTruthy();
  });
});
