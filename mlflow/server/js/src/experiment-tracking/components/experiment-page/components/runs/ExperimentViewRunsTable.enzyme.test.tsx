import { mount } from 'enzyme';
import { EXPERIMENT_RUNS_MOCK_STORE } from '../../fixtures/experiment-runs.fixtures';
import { ExperimentPageViewState } from '../../models/ExperimentPageViewState';
import { useRunsColumnDefinitions } from '../../utils/experimentPage.column-utils';
import { ExperimentViewRunsTable, ExperimentViewRunsTableProps } from './ExperimentViewRunsTable';
import { MemoryRouter } from '../../../../../common/utils/RoutingUtils';
import { createExperimentPageUIState } from '../../models/ExperimentPageUIState';
import { createExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import { MockedReduxStoreProvider } from '../../../../../common/utils/TestUtils';
import { makeCanonicalSortKey } from '../../utils/experimentPage.common-utils';
import { COLUMN_TYPES } from '../../../../constants';

/**
 * Mock all expensive utility functions
 */
jest.mock('../../utils/experimentPage.column-utils', () => ({
  ...jest.requireActual<typeof import('../../utils/experimentPage.column-utils')>(
    '../../utils/experimentPage.column-utils',
  ),
  useRunsColumnDefinitions: jest.fn(() => []),
  makeCanonicalSortKey: jest.requireActual('../../utils/experimentPage.common-utils').makeCanonicalSortKey,
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
  ...jest.requireActual<typeof import('react-intl')>('react-intl'),
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
    searchFacetsState: Object.assign(createExperimentPageSearchFacetsState(), {
      runsPinned: ['experiment123456789_run1'],
    }),
    viewState: new ExperimentPageViewState(),
    uiState: createExperimentPageUIState(),
    updateViewState() {},
    loadMoreRunsFunc: jest.fn(),
    expandRows: false,
    compareRunsMode: 'TABLE',
  };

  const createWrapper = (additionalProps: Partial<ExperimentViewRunsTableProps> = {}) =>
    mount(<ExperimentViewRunsTable {...defaultProps} {...additionalProps} />, {
      wrappingComponent: ({ children }: React.PropsWithChildren<unknown>) => (
        <MemoryRouter>
          <MockedReduxStoreProvider>{children}</MockedReduxStoreProvider>
        </MemoryRouter>
      ),
    });

  const createLargeDatasetProps = (selectedKey: string, columnType: string) => {
    const largeParamKeyList = Array.from({ length: 400 }, (_, i) => `p${i}`);
    const largeMetricKeyList = Array.from({ length: 400 }, (_, i) => `m${i}`);
    const largeTags: Record<string, { key: string; value: string }> = {};

    // Add the selected key to the appropriate list
    if (columnType === COLUMN_TYPES.PARAMS) {
      largeParamKeyList.push(selectedKey);
    } else if (columnType === COLUMN_TYPES.METRICS) {
      largeMetricKeyList.push(selectedKey);
    } else if (columnType === COLUMN_TYPES.TAGS) {
      largeTags[selectedKey] = { key: selectedKey, value: 'testvalue' };
    }

    // Create enough tags to exceed threshold
    for (let i = 0; i < 201; i++) {
      largeTags[`tag${i}`] = { key: `tag${i}`, value: `value${i}` };
    }

    return {
      runsData: {
        ...defaultProps.runsData,
        paramKeyList: largeParamKeyList,
        metricKeyList: largeMetricKeyList,
        tagsList: [largeTags],
      },
      uiState: Object.assign(createExperimentPageUIState(), {
        selectedColumns: [makeCanonicalSortKey(columnType, selectedKey)],
      }),
    };
  };

  test('should properly call creating column definitions function', () => {
    createWrapper();
    expect(useRunsColumnDefinitions).toHaveBeenCalledWith(
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

  test('should pass selected tag columns to column definitions', () => {
    const tagKey = mockTagKeys[0];
    createWrapper({
      uiState: Object.assign(createExperimentPageUIState(), {
        selectedColumns: [makeCanonicalSortKey(COLUMN_TYPES.TAGS, tagKey)],
      }),
    });
    expect(useRunsColumnDefinitions).toHaveBeenCalledWith(expect.objectContaining({ tagKeyList: mockTagKeys }));
  });

  test('should filter tag columns when shouldOptimize is true', () => {
    const tagKey = 'testtag1';
    createWrapper(createLargeDatasetProps(tagKey, COLUMN_TYPES.TAGS));

    const lastCall = (useRunsColumnDefinitions as jest.Mock).mock.calls.slice(-1)[0][0];
    expect(lastCall.tagKeyList).toEqual([tagKey]);
  });

  test('should filter metric columns when shouldOptimize is true', () => {
    const metricKey = 'testmetric1';
    createWrapper(createLargeDatasetProps(metricKey, COLUMN_TYPES.METRICS));

    const lastCall = (useRunsColumnDefinitions as jest.Mock).mock.calls.slice(-1)[0][0];
    expect(lastCall.metricKeyList).toEqual([metricKey]);
  });

  test('should filter param columns when shouldOptimize is true', () => {
    const paramKey = 'testparam1';
    createWrapper(createLargeDatasetProps(paramKey, COLUMN_TYPES.PARAMS));

    const lastCall = (useRunsColumnDefinitions as jest.Mock).mock.calls.slice(-1)[0][0];
    expect(lastCall.paramKeyList).toEqual([paramKey]);
  });

  test('should properly generate new column data on the new runs data', () => {
    const wrapper = createWrapper();

    // Assert that we're not calling for generating columns
    // while having "newparam" parameter
    expect(useRunsColumnDefinitions).not.toHaveBeenCalledWith(
      expect.objectContaining({
        paramKeyList: ['p1', 'p2', 'p3', 'newparam'],
      }),
    );

    // Update the param key list with "newparam" as a new entry
    wrapper.setProps({
      runsData: { ...defaultProps.runsData, paramKeyList: ['p1', 'p2', 'p3', 'newparam'] },
    });

    // Assert that "newparam" parameter is being included in calls
    // for new columns - but only if it's in the selected columns
    expect(useRunsColumnDefinitions).toHaveBeenCalledWith(
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
      searchFacetsState: Object.assign(createExperimentPageSearchFacetsState(), {
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
    expect(mockGridApi.setRowData).not.toHaveBeenCalledWith(
      expect.arrayContaining([expect.objectContaining({ isLoadMoreRow: true })]),
    );

    // Change the more runs flag to true
    containingExperimentsWrapper.setProps({
      moreRunsAvailable: true,
    });

    // Assert "load more" row being added to payload
    expect(mockGridApi.setRowData).toHaveBeenCalledWith(
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
      searchFacetsState: Object.assign(createExperimentPageSearchFacetsState(), {
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
      'attributes.`Description`',
    ];

    simpleExperimentsWrapper.setProps({
      runsData: {
        ...defaultProps.runsData,
        // Set new params and metrics
        paramKeyList: ['p1'],
        metricKeyList: ['m1'],
      },
      searchFacetsState: Object.assign(createExperimentPageSearchFacetsState(), {
        // Exhaust all possible columns
        selectedColumns: newSelectedColumns,
      }),
      uiState: Object.assign(createExperimentPageUIState(), {
        // Exhaust all possible columns
        selectedColumns: newSelectedColumns,
      }),
    });

    // With the selected columns including 'params.`p1`' and 'metrics.`m1`',
    // the filtered paramKeyList and metricKeyList should now include these values
    expect(useRunsColumnDefinitions).toHaveBeenCalledWith(
      expect.objectContaining({
        paramKeyList: ['p1'],
        metricKeyList: ['m1'],
      }),
    );

    // Assert "show more columns" CTA button not being displayed anymore
    expect(simpleExperimentsWrapper.find('ExperimentViewRunsTableAddColumnCTA').length).toBe(0);
  });
});
