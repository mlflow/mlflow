import { DesignSystemProvider } from '@databricks/design-system';
import { useState } from 'react';
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

const doStatefulMock = () => {
  const mockUpdateSearchFacets = jest.fn();
  let currentState: any;

  const getCurrentState = () => currentState;

  const Component = () => {
    const [searchFacetsState, setSearchFacetsState] = useState<SearchExperimentRunsFacetsState>(
      new SearchExperimentRunsFacetsState(),
    );

    currentState = searchFacetsState;

    const updateSearchFacets = (updatedFilterState: Partial<SearchExperimentRunsFacetsState>) => {
      mockUpdateSearchFacets(updatedFilterState);
      setSearchFacetsState((s: any) => ({ ...s, ...updatedFilterState }));
    };

    const props = {
      runsData: MOCK_RUNS_DATA,
      updateSearchFacets,
      searchFacetsState,
      sortOptions: [],
      viewState: {} as any,
      updateViewState: () => {},
    } as any;
    return (
      <DesignSystemProvider>
        <IntlProvider locale='en'>
          <ExperimentViewRunsControlsFilters {...props} />
        </IntlProvider>
      </DesignSystemProvider>
    );
  };
  return {
    wrapper: mountWithIntl(<Component />),
    mockUpdateSearchFacets,
    getCurrentState,
  };
};

describe('ExperimentViewRunsControlsFilters', () => {
  test('should render with given search facets model properly', () => {
    const searchFacetsState = new SearchExperimentRunsFacetsState();

    const wrapper = doSimpleMock({
      runsData: MOCK_RUNS_DATA,
      updateSearchFacets: jest.fn(),
      sortOptions: [],
      onDownloadCsv: () => {},
      viewState: {} as any,
      updateViewState: () => {},
      searchFacetsState,
    });

    expect(wrapper).toBeTruthy();
  });
  test('should update search facets model when searching by query', () => {
    const searchFacetsState = new SearchExperimentRunsFacetsState();

    const props = {
      runsData: MOCK_RUNS_DATA,
      updateSearchFacets: jest.fn(),
      viewState: {} as any,
      sortOptions: [],
      onDownloadCsv: () => {},
      updateViewState: () => {},
      searchFacetsState,
    };

    const wrapper = doSimpleMock(props);

    const searchInput = wrapper.find("input[data-test-id='search-box']");
    searchInput.simulate('change', { target: { value: 'test-query' } });
    searchInput.simulate('keydown', { key: 'Enter' });

    expect(props.updateSearchFacets).toBeCalledWith({ searchFilter: 'test-query' });
  });

  test('should update search facets model and properly clear filters afterwards', () => {
    const { wrapper, getCurrentState } = doStatefulMock();

    wrapper
      .find("input[data-test-id='search-box']")
      .simulate('change', { target: { value: 'test-query' } });
    wrapper.find("input[data-test-id='search-box']").simulate('keydown', { key: 'Enter' });

    expect(getCurrentState()).toMatchObject(
      expect.objectContaining({ searchFilter: 'test-query' }),
    );

    wrapper.find("input[data-test-id='search-box']").simulate('click');
    wrapper.find("button[data-test-id='clear-button']").simulate('click');

    expect(getCurrentState()).toMatchObject(new SearchExperimentRunsFacetsState());
  });
});
