import type { ReactWrapper } from 'enzyme';
import { MockedReduxStoreProvider } from '../../../../../common/utils/TestUtils';
import { mountWithIntl } from 'common/utils/TestUtils.enzyme';
import { GetExperimentRunsContext } from '../../contexts/GetExperimentRunsContext';
import { EXPERIMENT_RUNS_MOCK_STORE } from '../../fixtures/experiment-runs.fixtures';
import { SearchExperimentRunsFacetsState } from '../../models/SearchExperimentRunsFacetsState';
import type { GetExperimentRunsContextType } from '../../contexts/GetExperimentRunsContext';
import { ExperimentViewRunsImpl as ExperimentViewRuns, ExperimentViewRunsProps } from './ExperimentViewRuns';
import { MemoryRouter } from '../../../../../common/utils/RoutingUtils';
import { createExperimentPageUIStateV2 } from '../../models/ExperimentPageUIStateV2';

/**
 * Mock all expensive utility functions
 */
const mockPrepareRunsGridData = jest.fn();
jest.mock('../../utils/experimentPage.row-utils', () => ({
  ...jest.requireActual('../../utils/experimentPage.row-utils'),
  prepareRunsGridData: (...params: any) => mockPrepareRunsGridData(...params),
  useExperimentRunRows: (...params: any) => mockPrepareRunsGridData(...params),
}));

jest.mock('../../hooks/useCreateNewRun', () => ({
  CreateNewRunContextProvider: ({ children }: any) => <div>{children}</div>,
}));
jest.mock('./ExperimentViewRunsControls', () => ({
  ExperimentViewRunsControls: () => <div />,
}));
jest.mock('./ExperimentViewRunsTable', () => ({
  ExperimentViewRunsTable: ({ loadMoreRunsFunc }: { loadMoreRunsFunc: () => void }) => (
    <div>
      <button data-testid="load-more" onClick={loadMoreRunsFunc} />
    </div>
  ),
}));

/**
 * Mock all external components for performant mount() usage
 */
jest.mock('./ExperimentViewRunsEmptyTable', () => ({
  ExperimentViewRunsEmptyTable: () => <div />,
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

const mockedShowNotification = jest.fn();
jest.mock('../../hooks/useFetchedRunsNotification', () => {
  return {
    useFetchedRunsNotification: () => mockedShowNotification,
  };
});

const mockTagKeys = Object.keys(EXPERIMENT_RUNS_MOCK_STORE.entities.tagsByRunUuid['experiment123456789_run1']);

describe('ExperimentViewRuns', () => {
  let contextValue: Partial<GetExperimentRunsContextType> = {};
  const loadMoreRunsMockFn = jest.fn();

  beforeAll(() => {
    jest.useFakeTimers();
    jest.setSystemTime(new Date(2022, 0, 1));
  });

  afterAll(() => {
    jest.useRealTimers();
  });

  beforeEach(() => {
    mockedShowNotification.mockClear();
    mockPrepareRunsGridData.mockClear();
    mockPrepareRunsGridData.mockImplementation(() => []);

    contextValue = {
      actions: {},
      searchFacetsState: Object.assign(new SearchExperimentRunsFacetsState(), {
        runsPinned: ['experiment123456789_run1'],
      }),
      fetchExperimentRuns: jest.fn(),
      updateSearchFacets: jest.fn(),
      loadMoreRuns: loadMoreRunsMockFn,
      isPristine: jest.fn(),
    } as any;
  });

  const defaultProps: ExperimentViewRunsProps = {
    experiments: [EXPERIMENT_RUNS_MOCK_STORE.entities.experimentsById['123456789']],
    runsData: {
      paramKeyList: ['p1', 'p2', 'p3'],
      metricKeyList: ['m1', 'm2', 'm3'],
      modelVersionsByRunUuid: {},
      runUuidsMatchingFilter: ['experiment123456789_run1'],
      tagsList: [EXPERIMENT_RUNS_MOCK_STORE.entities.tagsByRunUuid['experiment123456789_run1']],
      runInfos: [EXPERIMENT_RUNS_MOCK_STORE.entities.runInfosByUuid['experiment123456789_run1']],
      paramsList: [[{ key: 'p1', value: 'pv1' }]],
      metricsList: [[{ key: 'm1', value: 'mv1' }]],
      datasetsList: [[{ dataset: { digest: 'ab12', name: 'dataset_name' } }]],
    } as any,
    isLoading: false,
    searchFacetsState: new SearchExperimentRunsFacetsState(),
    uiState: Object.assign(createExperimentPageUIStateV2(), {
      runsPinned: ['experiment123456789_run1'],
    }),
    isLoadingRuns: false,
    isPristine: () => false,
    loadMoreRuns: loadMoreRunsMockFn,
    refreshRuns: jest.fn(),
    requestError: null,
    moreRunsAvailable: false,
  };
  const ProxyComponent = (additionalProps: Partial<ExperimentViewRunsProps> = {}) => (
    <MemoryRouter>
      <MockedReduxStoreProvider state={{ entities: {} }}>
        <GetExperimentRunsContext.Provider value={contextValue as any}>
          <ExperimentViewRuns {...defaultProps} {...additionalProps} />
        </GetExperimentRunsContext.Provider>
      </MockedReduxStoreProvider>
    </MemoryRouter>
  );

  const createWrapper = (additionalProps: Partial<ExperimentViewRunsProps> = {}) =>
    mountWithIntl(<ProxyComponent {...additionalProps} />);

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
  });

  test('displays "(...) fetched more runs" notification when necessary', async () => {
    contextValue.moreRunsAvailable = true;
    contextValue.isLoadingRuns = false;

    loadMoreRunsMockFn.mockResolvedValue([{ info: { run_uuid: 'new' } }]);
    const wrapper = createWrapper({
      moreRunsAvailable: true,
      isLoadingRuns: false,
    });

    const button = wrapper.find("[data-testid='load-more']").first();

    button.simulate('click');
    await loadMoreRunsMockFn();

    expect(mockedShowNotification).toBeCalledWith(
      [{ info: { run_uuid: 'new' } }],
      [EXPERIMENT_RUNS_MOCK_STORE.entities.runInfosByUuid['experiment123456789_run1']],
    );
  });
});
