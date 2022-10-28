import { mount } from 'enzyme';
import { EXPERIMENT_RUNS_MOCK_STORE } from '../../fixtures/experiment-runs.fixtures';
import { SearchExperimentRunsFacetsState } from '../../models/SearchExperimentRunsFacetsState';
import { SearchExperimentRunsViewState } from '../../models/SearchExperimentRunsViewState';
import { useRunsColumnDefinitions } from '../../utils/experimentPage.column-utils';
import { ExperimentViewRunsTable, ExperimentViewRunsTableProps } from './ExperimentViewRunsTable';

/**
 * Mock all expensive utility functions
 */
jest.mock('../../utils/experimentPage.column-utils', () => ({
  ...jest.requireActual('../../utils/experimentPage.column-utils'),
  useRunsColumnDefinitions: jest.fn().mockImplementation(() => []),
}));

const mockPrepareRunsGridData = jest.fn();
jest.mock('../../utils/experimentPage.row-utils', () => ({
  ...jest.requireActual('../../utils/experimentPage.row-utils'),
  prepareRunsGridData: (...params: any) => mockPrepareRunsGridData(...params),
}));

/**
 * Mock all external components for performant mount() usage
 */
jest.mock('../../../../../common/components/ExperimentRunsTableEmptyOverlay', () => ({
  ExperimentRunsTableEmptyOverlay: () => <div />,
}));

jest.mock('../../../../../common/components/ag-grid/AgGridLoader', () => {
  const apiMock = {
    showLoadingOverlay: jest.fn(),
    hideOverlay: jest.fn(),
    setRowData: jest.fn(),
  };
  const columnApiMock = {};
  return {
    MLFlowAgGridLoader: ({ onGridReady }: any) => {
      onGridReady({
        api: apiMock,
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

const mockTagKeys = Object.keys(
  EXPERIMENT_RUNS_MOCK_STORE.entities.tagsByRunUuid['experiment123456789_run1'],
);

describe('ExperimentViewRunsTable', () => {
  beforeAll(() => {
    jest.useFakeTimers();
    jest.setSystemTime(new Date(2022, 0, 1));
  });

  afterAll(() => {
    jest.useRealTimers();
  });

  beforeEach(() => {
    mockPrepareRunsGridData.mockClear();
    mockPrepareRunsGridData.mockImplementation(() => []);
  });

  const defaultProps: ExperimentViewRunsTableProps = {
    experiments: [EXPERIMENT_RUNS_MOCK_STORE.entities.experimentsById['123456789']],
    isLoading: false,
    onAddColumnClicked() {},
    runsData: {
      paramKeyList: ['p1', 'p2', 'p3'],
      metricKeyList: ['m1', 'm2', 'm3'],
      tagsList: [EXPERIMENT_RUNS_MOCK_STORE.entities.tagsByRunUuid['experiment123456789_run1']],
      runInfos: [EXPERIMENT_RUNS_MOCK_STORE.entities.runInfosByUuid['experiment123456789_run1']],
      paramsList: [[{ key: 'p1', value: 'pv1' }]],
      metricsList: [[{ key: 'm1', value: 'mv1' }]],
    } as any,
    searchFacetsState: Object.assign(new SearchExperimentRunsFacetsState(), {
      runsPinned: ['experiment123456789_run1'],
    }),
    viewState: new SearchExperimentRunsViewState(),
    updateSearchFacets() {},
    updateViewState() {},
  };
  const createWrapper = (additionalProps: Partial<ExperimentViewRunsTableProps> = {}) =>
    mount(<ExperimentViewRunsTable {...defaultProps} {...additionalProps} />);

  test('should properly call creating column definitions function', () => {
    createWrapper();
    expect(useRunsColumnDefinitions).toBeCalledWith(
      expect.objectContaining({
        searchFacetsState: expect.anything(),
        compareExperiments: false,
        metricKeyList: ['m1', 'm2', 'm3'],
        paramKeyList: ['p1', 'p2', 'p3'],
        tagKeyList: mockTagKeys,
        columnApi: expect.anything(),
      }),
    );
  });

  test('should properly call getting row data function', () => {
    createWrapper();
    expect(mockPrepareRunsGridData).toBeCalledWith(
      expect.objectContaining({
        metricKeyList: ['m1', 'm2', 'm3'],
        paramKeyList: ['p1', 'p2', 'p3'],
        tagKeyList: mockTagKeys,
        runsPinned: ['experiment123456789_run1'],
        referenceTime: new Date(),
        nestChildren: true,
      }),
    );
  });

  test('should properly react to the new runs data', () => {
    const wrapper = createWrapper();

    // Assert that we're not calling for generating columns/rows
    // while having "newparam" parameter
    expect(mockPrepareRunsGridData).not.toBeCalledWith(
      expect.objectContaining({
        paramKeyList: ['p1', 'p2', 'p3', 'newparam'],
      }),
    );
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
    // for new columns and rows
    expect(mockPrepareRunsGridData).toBeCalledWith(
      expect.objectContaining({
        paramKeyList: ['p1', 'p2', 'p3', 'newparam'],
      }),
    );
    expect(useRunsColumnDefinitions).toBeCalledWith(
      expect.objectContaining({
        paramKeyList: ['p1', 'p2', 'p3', 'newparam'],
      }),
    );
  });

  test('should hide no data overlay when necessary', () => {
    // Prepare a runs grid data with an empty set
    mockPrepareRunsGridData.mockImplementation(() => []);
    const emptyExperimentsWrapper = createWrapper();

    // Assert empty overlay being displayed
    expect(emptyExperimentsWrapper.find('ExperimentRunsTableEmptyOverlay').length).toBe(1);
  });

  test('should display no data overlay when necessary', () => {
    // Prepare a runs grid data with a non-empty set
    mockPrepareRunsGridData.mockImplementation(() => [{ abc: 123 }]);
    const containingExperimentsWrapper = createWrapper();

    // Assert empty overlay being not displayed
    expect(containingExperimentsWrapper.find('ExperimentRunsTableEmptyOverlay').length).toBe(0);
  });

  test('should hide column CTA when all columns have been selected', () => {
    // Prepare a runs grid data with a default metrics/params set and no initially selected columns
    const simpleExperimentsWrapper = createWrapper();

    // Assert "add params/metrics" CTA button being displayed
    expect(simpleExperimentsWrapper.find('ExperimentViewRunsTableAddColumnCTA').length).toBe(1);

    simpleExperimentsWrapper.setProps({
      runsData: {
        ...defaultProps.runsData,
        // Set new params and metrics
        paramKeyList: ['p1'],
        metricKeyList: ['m1'],
      },
      searchFacetsState: Object.assign(new SearchExperimentRunsFacetsState(), {
        // Exhaust all possible columns
        selectedColumns: [
          'params.`p1`',
          'metrics.`m1`',
          'tags.`testtag1`',
          'tags.`testtag2`',
          'attributes.`User`',
          'attributes.`Source`',
          'attributes.`Version`',
          'attributes.`Models`',
        ],
      }),
    });

    // Assert "add params/metrics" CTA button not being displayed anymore
    expect(simpleExperimentsWrapper.find('ExperimentViewRunsTableAddColumnCTA').length).toBe(0);
  });
});
