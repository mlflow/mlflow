import { mount } from 'enzyme';
import { EXPERIMENT_RUNS_MOCK_STORE } from '../../fixtures/experiment-runs.fixtures';
import { SearchExperimentRunsFacetsState } from '../../models/SearchExperimentRunsFacetsState';
import { SearchExperimentRunsViewState } from '../../models/SearchExperimentRunsViewState';
import { useRunsColumnDefinitions } from '../../utils/experimentPage.column-utils';
import { ExperimentViewRunsTable, ExperimentViewRunsTableProps } from './ExperimentViewRunsTable';
import { MemoryRouter } from '../../../../../common/utils/RoutingUtils';
import { createExperimentPageUIStateV2 } from '../../models/ExperimentPageUIStateV2';

/**
 * Mock all expensive utility functions
 */
jest.mock('../../utils/experimentPage.column-utils', () => ({
  ...jest.requireActual('../../utils/experimentPage.column-utils'),
  useRunsColumnDefinitions: jest.fn().mockImplementation(() => []),
}));

/**
 * Mock all external components for performant mount() usage
 */
jest.mock('./ExperimentViewRunsEmptyTable', () => ({
  ExperimentViewRunsEmptyTable: () => <div />,
}));

/**
 * Mock all external components for performant mount() usage
 */
jest.mock('./ExperimentViewRunsTableStatusBar', () => ({
  ExperimentViewRunsTableStatusBar: () => <div />,
}));

// ExperimentViewRunsTableAddColumnCTA isn't supported in this test as it uses ResizeObserver
jest.mock('./ExperimentViewRunsTableAddColumnCTA', () => ({
  ExperimentViewRunsTableAddColumnCTA: () => null,
}));

const mockGridApi = {
  showLoadingOverlay: jest.fn(),
  hideOverlay: jest.fn(),
  setRowData: jest.fn(),
  resetRowHeights: jest.fn(),
};

jest.mock('../../../../../common/components/ag-grid/AgGridLoader', () => {
  const columnApiMock = {};
  return {
    MLFlowAgGridLoader: ({ onGridReady }: any) => {
      onGridReady({
        api: mockGridApi,
        columnApi: columnApiMock,
      });
      return <div />;
    },
  };
});

/**
 * Mock <FormattedMessage /> instead of providing intl context to make
 * settings enzyme wrapper's props prossible
 */
jest.mock('react-intl', () => ({
  ...jest.requireActual('react-intl'),
  FormattedMessage: () => <div />,
}));

const mockTagKeys = Object.keys(EXPERIMENT_RUNS_MOCK_STORE.entities.tagsByRunUuid['experiment123456789_run1']);

describe('ExperimentViewRunsTable', () => {
  beforeAll(() => {
    jest.useFakeTimers();
    jest.setSystemTime(new Date(2022, 0, 1));
  });

  afterAll(() => {
    jest.useRealTimers();
  });

  const defaultProps: ExperimentViewRunsTableProps = {
    experiments: [EXPERIMENT_RUNS_MOCK_STORE.entities.experimentsById['123456789']],
    moreRunsAvailable: false,
    isLoading: false,
    onAddColumnClicked() {},
    runsData: {
      paramKeyList: ['p1', 'p2', 'p3'],
      metricKeyList: ['m1', 'm2', 'm3'],
      tagsList: [EXPERIMENT_RUNS_MOCK_STORE.entities.tagsByRunUuid['experiment123456789_run1']],
      runInfos: [EXPERIMENT_RUNS_MOCK_STORE.entities.runInfosByUuid['experiment123456789_run1']],
      paramsList: [[{ key: 'p1', value: 'pv1' }]],
      metricsList: [[{ key: 'm1', value: 'mv1' }]],
      runUuidsMatchingFilter: ['experiment123456789_run1'],
    } as any,
    rowsData: [{ runUuid: 'experiment123456789_run1' } as any],
    searchFacetsState: Object.assign(new SearchExperimentRunsFacetsState(), {
      runsPinned: ['experiment123456789_run1'],
    }),
    viewState: new SearchExperimentRunsViewState(),
    uiState: createExperimentPageUIStateV2(),
    updateSearchFacets() {},
    updateViewState() {},
    loadMoreRunsFunc: jest.fn(),
    expandRows: false,
  };

  const createWrapper = (additionalProps: Partial<ExperimentViewRunsTableProps> = {}) =>
    mount(<ExperimentViewRunsTable {...defaultProps} {...additionalProps} />, {
      wrappingComponent: MemoryRouter,
    });

  test('should properly call creating column definitions function', () => {
    createWrapper();
    expect(useRunsColumnDefinitions).toBeCalledWith(
      expect.objectContaining({
        selectedColumns: expect.anything(),
        compareExperiments: false,
        metricKeyList: ['m1', 'm2', 'm3'],
        paramKeyList: ['p1', 'p2', 'p3'],
        tagKeyList: mockTagKeys,
        columnApi: expect.anything(),
      }),
    );
  });

  test('should properly generate new column data on the new runs data', () => {
    const wrapper = createWrapper();

    // Assert that we're not calling for generating columns
    // while having "newparam" parameter
    expect(useRunsColumnDefinitions).not.toBeCalledWith(
      expect.objectContaining({
        paramKeyList: ['p1', 'p2', 'p3', 'newparam'],
      }),
    );

    // Update the param key list with "newparam" as a new entry
    wrapper.setProps({
      runsData: { ...defaultProps.runsData, paramKeyList: ['p1', 'p2', 'p3', 'newparam'] },
    });

    // Assert that "newparam" parameter is being included in calls
    // for new columns
    expect(useRunsColumnDefinitions).toBeCalledWith(
      expect.objectContaining({
        paramKeyList: ['p1', 'p2', 'p3', 'newparam'],
      }),
    );
  });

  test('should display no data overlay with proper configuration and only when necessary', () => {
    // Prepare a runs grid data with an empty set
    const emptyExperimentsWrapper = createWrapper({ rowsData: [] });

    // Assert empty overlay being displayed and indicating that runs are *not* filtered
    expect(emptyExperimentsWrapper.find('ExperimentViewRunsEmptyTable').length).toBe(1);
    expect(emptyExperimentsWrapper.find('ExperimentViewRunsEmptyTable').prop('isFiltered')).toBe(false);

    // Set up some filter
    emptyExperimentsWrapper.setProps({
      searchFacetsState: Object.assign(new SearchExperimentRunsFacetsState(), {
        searchFilter: 'something',
      }),
    });

    // Assert empty overlay being displayed and indicating that runs *are* filtered
    expect(emptyExperimentsWrapper.find('ExperimentViewRunsEmptyTable').prop('isFiltered')).toBe(true);
  });

  test('should hide no data overlay when necessary', () => {
    // Prepare a runs grid data with a non-empty set
    const containingExperimentsWrapper = createWrapper();

    // Assert empty overlay being not displayed
    expect(containingExperimentsWrapper.find('ExperimentViewRunsEmptyTable').length).toBe(0);
  });

  test('should properly show "load more" button when necessary', () => {
    mockGridApi.setRowData.mockClear();

    // Prepare a runs grid data with a non-empty set
    const containingExperimentsWrapper = createWrapper({ moreRunsAvailable: false });

    // Assert "load more" row not being sent to agGrid
    expect(mockGridApi.setRowData).not.toBeCalledWith(
      expect.arrayContaining([expect.objectContaining({ isLoadMoreRow: true })]),
    );

    // Change the more runs flag to true
    containingExperimentsWrapper.setProps({
      moreRunsAvailable: true,
    });

    // Assert "load more" row being added to payload
    expect(mockGridApi.setRowData).toBeCalledWith(
      expect.arrayContaining([expect.objectContaining({ isLoadMoreRow: true })]),
    );
  });

  test('should display proper status bar with runs length', () => {
    const wrapper = createWrapper();

    // Find status bar and expect it to display 1 run
    expect(wrapper.find('ExperimentViewRunsTableStatusBar').prop('allRunsCount')).toEqual(1);

    // Change the filtered run set so it mimics the scenario where used has unpinned the row
    wrapper.setProps({
      runsData: { ...defaultProps.runsData, runUuidsMatchingFilter: [] },
      searchFacetsState: Object.assign(new SearchExperimentRunsFacetsState(), {
        runsPinned: [],
      }),
    });

    // Find status bar and expect it to display 0 runs
    expect(wrapper.find('ExperimentViewRunsTableStatusBar').prop('allRunsCount')).toEqual(0);
    expect(wrapper.find('ExperimentViewRunsTableStatusBar').prop('isLoading')).toEqual(false);

    // Set loading flag to true
    wrapper.setProps({ isLoading: true });

    // Expect status bar to display spinner as well
    expect(wrapper.find('ExperimentViewRunsTableStatusBar').prop('isLoading')).toEqual(true);
  });

  test('should hide column CTA when all columns have been selected', () => {
    // Prepare a runs grid data with a default metrics/params set and no initially selected columns
    const simpleExperimentsWrapper = createWrapper();

    // Assert "add params/metrics" CTA button being displayed
    expect(simpleExperimentsWrapper.find('ExperimentViewRunsTableAddColumnCTA').length).toBe(1);

    const newSelectedColumns = [
      'params.`p1`',
      'metrics.`m1`',
      'tags.`testtag1`',
      'tags.`testtag2`',
      'attributes.`User`',
      'attributes.`Source`',
      'attributes.`Version`',
      'attributes.`Models`',
      'attributes.`Dataset`',
    ];

    simpleExperimentsWrapper.setProps({
      runsData: {
        ...defaultProps.runsData,
        // Set new params and metrics
        paramKeyList: ['p1'],
        metricKeyList: ['m1'],
      },
      searchFacetsState: Object.assign(new SearchExperimentRunsFacetsState(), {
        // Exhaust all possible columns
        selectedColumns: newSelectedColumns,
      }),
      uiState: Object.assign(createExperimentPageUIStateV2(), {
        // Exhaust all possible columns
        selectedColumns: newSelectedColumns,
      }),
    });

    // Assert "show more columns" CTA button not being displayed anymore
    expect(simpleExperimentsWrapper.find('ExperimentViewRunsTableAddColumnCTA').length).toBe(0);
  });
});
