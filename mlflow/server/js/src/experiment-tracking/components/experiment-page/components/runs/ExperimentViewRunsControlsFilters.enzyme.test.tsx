import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { BrowserRouter } from '../../../../../common/utils/RoutingUtils';
import { mountWithIntl } from 'common/utils/TestUtils.enzyme';
import { EXPERIMENT_RUNS_MOCK_STORE } from '../../fixtures/experiment-runs.fixtures';
import { SearchExperimentRunsFacetsState } from '../../models/SearchExperimentRunsFacetsState';
import { experimentRunsSelector } from '../../utils/experimentRuns.selector';
import {
  ExperimentViewRunsControlsFilters,
  ExperimentViewRunsControlsFiltersProps,
} from './ExperimentViewRunsControlsFilters';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';

jest.mock('./ExperimentViewRefreshButton', () => ({
  ExperimentViewRefreshButton: () => <div />,
}));

const mockRunsContext = {
  updateSearchFacets: jest.fn(),
};

jest.mock('../../hooks/useFetchExperimentRuns', () => ({
  useFetchExperimentRuns: () => mockRunsContext,
}));

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
    const searchFacetsState = new SearchExperimentRunsFacetsState();

    const wrapper = doSimpleMock({
      runsData: MOCK_RUNS_DATA,
      updateSearchFacets: jest.fn(),
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
