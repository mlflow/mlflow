import { MockedReduxStoreProvider } from '../../../../../common/utils/TestUtils';
import { EXPERIMENT_RUNS_MOCK_STORE } from '../../fixtures/experiment-runs.fixtures';
import type { ExperimentViewRunsProps } from './ExperimentViewRuns';
import { ExperimentViewRuns } from './ExperimentViewRuns';
import { MemoryRouter } from '../../../../../common/utils/RoutingUtils';
import { createExperimentPageUIState } from '../../models/ExperimentPageUIState';
import { createExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { render, screen, waitFor } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { setupServer } from '../../../../../common/utils/setup-msw';
import { rest } from 'msw';
import userEvent from '@testing-library/user-event';
import { useFetchedRunsNotification } from '../../hooks/useFetchedRunsNotification';
import { useExperimentRunRows } from '../../utils/experimentPage.row-utils';
import { DesignSystemProvider } from '@databricks/design-system';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(90000); // Larger timeout for integration testing (table rendering)

// Rendering ag-grid table takes a lot of resources and time, we increase waitFor()'s timeout from default 5000 ms
const WAIT_FOR_TIMEOUT = 10_000;

// Enable feature flags
jest.mock('../../../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../../../common/utils/FeatureUtils')>(
    '../../../../../common/utils/FeatureUtils',
  ),
}));

// Mock rows preparation function to enable contract test
jest.mock('../../utils/experimentPage.row-utils', () => {
  const module = jest.requireActual<typeof import('../../utils/experimentPage.row-utils')>(
    '../../utils/experimentPage.row-utils',
  );
  return {
    ...module,
    useExperimentRunRows: jest.fn(module.useExperimentRunRows),
  };
});

// Mock less relevant, costly components
jest.mock('../../hooks/useCreateNewRun', () => ({
  CreateNewRunContextProvider: ({ children }: any) => <div>{children}</div>,
}));
jest.mock('./ExperimentViewRunsControls', () => ({
  ExperimentViewRunsControls: () => <div />,
}));
jest.mock('../../hooks/useFetchedRunsNotification', () => ({ useFetchedRunsNotification: jest.fn() }));

const mockTagKeys = Object.keys(EXPERIMENT_RUNS_MOCK_STORE.entities.tagsByRunUuid['experiment123456789_run1']);

describe('ExperimentViewRuns', () => {
  const server = setupServer();

  const loadMoreRunsMockFn = jest.fn();

  beforeAll(() => {
    jest.setSystemTime(new Date(2022, 0, 1));
    server.listen();
  });

  beforeEach(() => {
    jest.mocked(useExperimentRunRows).mockClear();
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
      experimentTags: {},
    } as any,
    isLoading: false,
    searchFacetsState: createExperimentPageSearchFacetsState(),
    uiState: Object.assign(createExperimentPageUIState(), {
      runsPinned: ['experiment123456789_run1'],
    }),
    isLoadingRuns: false,
    loadMoreRuns: loadMoreRunsMockFn,
    refreshRuns: jest.fn(),
    requestError: null,
    moreRunsAvailable: false,
  };

  const queryClient = new QueryClient();

  const renderTestComponent = (additionalProps: Partial<ExperimentViewRunsProps> = {}) => {
    return render(<ExperimentViewRuns {...defaultProps} {...additionalProps} />, {
      wrapper: ({ children }) => (
        <MemoryRouter>
          <IntlProvider locale="en">
            <QueryClientProvider client={queryClient}>
              <DesignSystemProvider>
                <MockedReduxStoreProvider
                  state={{
                    entities: {
                      modelVersionsByRunUuid: {},
                      colorByRunUuid: {},
                    },
                  }}
                >
                  {children}
                </MockedReduxStoreProvider>
              </DesignSystemProvider>
            </QueryClientProvider>
          </IntlProvider>
        </MemoryRouter>
      ),
    });
  };

  test('should render the table with all relevant data', async () => {
    renderTestComponent();

    // Assert cell with the name
    await waitFor(
      () => {
        expect(screen.getByRole('gridcell', { name: /experiment123456789_run1$/ })).toBeInTheDocument();
      },
      {
        timeout: WAIT_FOR_TIMEOUT,
      },
    );

    // Assert cell with the dataset
    expect(screen.getByRole('gridcell', { name: /dataset_name \(ab12\)/ })).toBeInTheDocument();
  });

  test('[contract] should properly call getting row data function', async () => {
    renderTestComponent();

    await waitFor(() => {
      expect(useExperimentRunRows).toHaveBeenCalledWith(
        expect.objectContaining({
          metricKeyList: ['m1', 'm2', 'm3'],
          paramKeyList: ['p1', 'p2', 'p3'],
          tagKeyList: mockTagKeys,
          runsPinned: ['experiment123456789_run1'],
          nestChildren: true,
        }),
      );
    });
  });

  test('should properly react to the new runs data', async () => {
    const { rerender } = renderTestComponent();

    await waitFor(() => {
      expect(useExperimentRunRows).toHaveBeenCalled();
    });
    // Assert that we're not calling for generating columns/rows
    // while having "newparam" parameter
    expect(useExperimentRunRows).not.toHaveBeenCalledWith(
      expect.objectContaining({
        paramKeyList: ['p1', 'p2', 'p3', 'newparam'],
      }),
    );

    // Update the param key list with "newparam" as a new entry
    rerender(
      <ExperimentViewRuns
        {...defaultProps}
        runsData={{ ...defaultProps.runsData, paramKeyList: ['p1', 'p2', 'p3', 'newparam'] }}
      />,
    );

    await waitFor(() => {
      // Assert that "newparam" parameter is being included in calls
      // for new columns and rows
      expect(useExperimentRunRows).toHaveBeenCalledWith(
        expect.objectContaining({
          paramKeyList: ['p1', 'p2', 'p3', 'newparam'],
        }),
      );
    });
  });

  test('displays "(...) fetched more runs" notification when necessary', async () => {
    const mockedShowNotification = jest.fn();
    jest.mocked(useFetchedRunsNotification).mockImplementation(() => mockedShowNotification);
    loadMoreRunsMockFn.mockResolvedValue([{ info: { runUuid: 'new' } }]);
    renderTestComponent({
      moreRunsAvailable: true,
      isLoadingRuns: false,
    });

    await waitFor(
      () => {
        expect(screen.getByRole('button', { name: 'Load more' })).toBeInTheDocument();
      },
      {
        timeout: WAIT_FOR_TIMEOUT,
      },
    );

    await userEvent.click(screen.getByRole('button', { name: 'Load more' }));

    await waitFor(() => {
      expect(mockedShowNotification).toHaveBeenLastCalledWith(
        [{ info: { runUuid: 'new' } }],
        [EXPERIMENT_RUNS_MOCK_STORE.entities.runInfosByUuid['experiment123456789_run1']],
      );
    });
  });
});
