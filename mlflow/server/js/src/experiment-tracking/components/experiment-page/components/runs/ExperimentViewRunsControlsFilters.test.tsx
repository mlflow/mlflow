import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { mountWithIntl } from '../../../../../common/utils/TestUtils';
import { EXPERIMENT_RUNS_MOCK_STORE } from '../../fixtures/experiment-runs.fixtures';
import { SearchExperimentRunsFacetsState } from '../../models/SearchExperimentRunsFacetsState';
import { experimentRunsSelector } from '../../utils/experimentRuns.selector';
import {
  ExperimentViewRunsControlsFilters,
  ExperimentViewRunsControlsFiltersProps,
} from './ExperimentViewRunsControlsFilters';

jest.mock('./ExperimentViewRefreshButton', () => ({
  ExperimentViewRefreshButton: () => <div />,
}));

const MOCK_EXPERIMENT = EXPERIMENT_RUNS_MOCK_STORE.entities.experimentsById['123456789'];

const MOCK_RUNS_DATA = experimentRunsSelector(EXPERIMENT_RUNS_MOCK_STORE, {
  experiments: [MOCK_EXPERIMENT],
});

jest.mock('./ExperimentViewRunsColumnSelector', () => ({
  ExperimentViewRunsColumnSelector: () => <div />,
}));

const doSimpleMock = (props: ExperimentViewRunsControlsFiltersProps) =>
  mountWithIntl(
    <DesignSystemProvider>
      <IntlProvider locale='en'>
        <ExperimentViewRunsControlsFilters {...props} />
      </IntlProvider>
    </DesignSystemProvider>,
  );

describe('ExperimentViewRunsControlsFilters', () => {
  test('should render with given search facets model properly', () => {
    const searchFacetsState = new SearchExperimentRunsFacetsState();

    const wrapper = doSimpleMock({
      runsData: MOCK_RUNS_DATA,
      updateSearchFacets: jest.fn(),
      sortOptions: [],
      onDownloadCsv: () => {},
      viewState: { runsSelected: {} } as any,
      updateViewState: () => {},
      searchFacetsState,
      requestError: null,
    });

    expect(wrapper).toBeTruthy();
  });
});
