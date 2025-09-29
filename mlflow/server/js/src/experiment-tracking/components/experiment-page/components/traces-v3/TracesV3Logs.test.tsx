import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TracesV3Logs } from './TracesV3Logs';
import { IntlProvider } from '@databricks/i18n';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { DesignSystemProvider } from '@databricks/design-system';
import {
  useMlflowTracesTableMetadata,
  useSearchMlflowTraces,
  useSelectedColumns,
  useFilters,
  useTableSort,
  GenAITracesTableProvider,
  REQUEST_TIME_COLUMN_ID,
  TracesTableColumnType,
  TracesTableColumnGroup,
} from '@databricks/web-shared/genai-traces-table';
import { useSetInitialTimeFilter } from './hooks/useSetInitialTimeFilter';
import { useDeleteTracesMutation } from '../../../evaluations/hooks/useDeleteTraces';
import { useEditExperimentTraceTags } from '../../../traces/hooks/useEditExperimentTraceTags';
import { useMarkdownConverter } from '@mlflow/mlflow/src/common/utils/MarkdownUtils';
import { GenericNetworkRequestError } from '@mlflow/mlflow/src/shared/web-shared/errors/PredefinedErrors';
import { TestRouter, testRoute, waitForRoutesToBeRendered } from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';

// Mock all external dependencies
jest.mock('@databricks/web-shared/genai-traces-table', () => {
  const actual = jest.requireActual<typeof import('@databricks/web-shared/genai-traces-table')>(
    '@databricks/web-shared/genai-traces-table',
  );
  return {
    ...actual,
    useMlflowTracesTableMetadata: jest.fn(),
    useSearchMlflowTraces: jest.fn(),
    useSelectedColumns: jest.fn(),
    useFilters: jest.fn(),
    useTableSort: jest.fn(),
    invalidateMlflowSearchTracesCache: jest.fn(),
  };
});

jest.mock('./hooks/useSetInitialTimeFilter', () => ({
  useSetInitialTimeFilter: jest.fn(),
}));

jest.mock('../../../evaluations/hooks/useDeleteTraces', () => ({
  useDeleteTracesMutation: jest.fn(),
}));

jest.mock('../../../traces/hooks/useEditExperimentTraceTags', () => ({
  useEditExperimentTraceTags: jest.fn(),
}));

jest.mock('@mlflow/mlflow/src/common/utils/MarkdownUtils', () => ({
  useMarkdownConverter: jest.fn(),
}));

jest.mock('@mlflow/mlflow/src/common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('@mlflow/mlflow/src/common/utils/FeatureUtils')>(
    '@mlflow/mlflow/src/common/utils/FeatureUtils',
  ),
  shouldEnableTagGrouping: jest.fn().mockReturnValue(true),
}));

jest.mock('@mlflow/mlflow/src/experiment-tracking/sdk/MlflowService', () => ({
  MlflowService: {
    getExperimentTraceInfoV3: jest.fn(),
    getExperimentTraceData: jest.fn(),
  },
}));

// Mock the empty state component to avoid deep dependency issues
jest.mock('./TracesV3EmptyState', () => ({
  TracesV3EmptyState: jest.fn(() => null),
}));

const renderComponent = (props = {}) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return render(
    <TestRouter
      routes={[
        testRoute(
          <IntlProvider locale="en">
            <QueryClientProvider client={queryClient}>
              <DesignSystemProvider>
                <GenAITracesTableProvider>
                  <TracesV3Logs experimentId="test-experiment" endpointName="test-endpoint" {...props} />
                </GenAITracesTableProvider>
              </DesignSystemProvider>
            </QueryClientProvider>
          </IntlProvider>,
        ),
      ]}
    />,
  );
};

describe('TracesV3Logs', () => {
  beforeEach(() => {
    // Default mock implementations
    jest.mocked(useMarkdownConverter).mockReturnValue((markdown?: string) => markdown || '');

    jest.mocked(useSelectedColumns).mockReturnValue({
      selectedColumns: [
        {
          id: REQUEST_TIME_COLUMN_ID,
          type: TracesTableColumnType.TRACE_INFO,
          group: TracesTableColumnGroup.INFO,
          label: 'Request Time',
        },
      ],
      toggleColumns: jest.fn(),
      setSelectedColumns: jest.fn(),
    });

    jest.mocked(useFilters).mockReturnValue([[], jest.fn()]);

    jest
      .mocked(useTableSort)
      .mockReturnValue([
        { key: REQUEST_TIME_COLUMN_ID, type: TracesTableColumnType.TRACE_INFO, asc: false },
        jest.fn(),
      ]);

    jest.mocked(useDeleteTracesMutation).mockReturnValue({
      mutateAsync: jest.fn(),
    } as any);

    jest.mocked(useEditExperimentTraceTags).mockReturnValue({
      showEditTagsModalForTrace: jest.fn(),
      EditTagsModal: <div>EditTagsModal</div>,
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  /**
   * Loading State Test Matrix
   *
   * This matrix covers all combinations of the 3 main loading states:
   * - isMetadataLoading (from useMlflowTracesTableMetadata)
   * - isInitialTimeFilterLoading (from useSetInitialTimeFilter)
   * - traceInfosLoading (from useSearchMlflowTraces)
   *
   * UI Components tracked:
   * - Toolbar: GenAITracesTableToolbar (always rendered, but column selector can be loading)
   * - Table: Either GenAITracesTableBodyContainer or ParagraphSkeleton components
   *
   * | isMetadataLoading | isInitialTimeFilterLoading | traceInfosLoading | UI in Loading State | UI Showing Data |
   * |-------------------|----------------------------|-------------------|---------------------|-----------------|
   * | false             | false                      | false             | None | Toolbar, Table |
   * | false             | false                      | true              | Table | Toolbar |
   * | false             | true                       | false             | Table | Toolbar |
   * | false             | true                       | true              | Table | Toolbar |
   * | true              | false                      | false             | Table | Toolbar (selecting a selector shows spinner) |
   * | true              | false                      | true              | Table | Toolbar (selecting a selector shows spinner) |
   * | true              | true                       | false             | Table | Toolbar (selecting a selector shows spinner) |
   * | true              | true                       | true              | Table | Toolbar (selecting a selector shows spinner) |
   *
   */

  const loadingStateMatrix = [
    {
      isMetadataLoading: false,
      isInitialTimeFilterLoading: false,
      traceInfosLoading: false,
      testName: 'all loading states false',
      uiInLoadingState: [],
      uiShowingData: ['toolbar', 'table'],
    },
    {
      isMetadataLoading: false,
      isInitialTimeFilterLoading: false,
      traceInfosLoading: true,
      testName: 'only traceInfosLoading true',
      uiInLoadingState: ['table'],
      uiShowingData: ['toolbar'],
    },
    {
      isMetadataLoading: false,
      isInitialTimeFilterLoading: true,
      traceInfosLoading: false,
      testName: 'only isInitialTimeFilterLoading true',
      uiInLoadingState: ['table'],
      uiShowingData: ['toolbar'],
    },
    {
      isMetadataLoading: false,
      isInitialTimeFilterLoading: true,
      traceInfosLoading: true,
      testName: 'isInitialTimeFilterLoading and traceInfosLoading true',
      uiInLoadingState: ['table'],
      uiShowingData: ['toolbar'],
    },
    {
      isMetadataLoading: true,
      isInitialTimeFilterLoading: false,
      traceInfosLoading: false,
      testName: 'only isMetadataLoading true',
      uiInLoadingState: ['table'],
      uiShowingData: ['toolbar (selecting a selector shows spinner)'],
    },
    {
      isMetadataLoading: true,
      isInitialTimeFilterLoading: false,
      traceInfosLoading: true,
      testName: 'isMetadataLoading and traceInfosLoading true',
      uiInLoadingState: ['table'],
      uiShowingData: ['toolbar (selecting a selector shows spinner)'],
    },
    {
      isMetadataLoading: true,
      isInitialTimeFilterLoading: true,
      traceInfosLoading: false,
      testName: 'isMetadataLoading and isInitialTimeFilterLoading true',
      uiInLoadingState: ['table'],
      uiShowingData: ['toolbar (selecting a selector shows spinner)'],
    },
    {
      isMetadataLoading: true,
      isInitialTimeFilterLoading: true,
      traceInfosLoading: true,
      testName: 'all loading states true',
      uiInLoadingState: ['table'],
      uiShowingData: ['toolbar (selecting a selector shows spinner)'],
    },
  ];

  describe('Error handling', () => {
    it('should display error in table when useMlflowTracesTableMetadata errors', async () => {
      const mockError = new GenericNetworkRequestError({ status: 500 }, new Error('Failed to fetch metadata'));

      jest.mocked(useMlflowTracesTableMetadata).mockReturnValue({
        assessmentInfos: [],
        allColumns: [],
        totalCount: 0,
        isLoading: false,
        error: mockError,
        isEmpty: false,
        tableFilterOptions: { source: [] },
        evaluatedTraces: [],
        otherEvaluatedTraces: [],
      });

      jest.mocked(useSetInitialTimeFilter).mockReturnValue({
        isInitialTimeFilterLoading: false,
      });

      jest.mocked(useSearchMlflowTraces).mockReturnValue({
        data: [],
        isLoading: false,
        isFetching: false,
        error: null,
      } as any);

      renderComponent();
      await waitForRoutesToBeRendered();

      // Verify error is displayed in the table body
      expect(screen.getByText('Fetching traces failed')).toBeInTheDocument();
      expect(screen.getByText('A network error occurred.')).toBeInTheDocument();
    });

    it('should display error in each selector when useMlflowTracesTableMetadata errors', async () => {
      const mockError = new GenericNetworkRequestError({ status: 500 }, new Error('Failed to fetch metadata'));

      jest.mocked(useMlflowTracesTableMetadata).mockReturnValue({
        assessmentInfos: [],
        allColumns: [],
        totalCount: 0,
        isLoading: false,
        error: mockError,
        isEmpty: false,
        tableFilterOptions: { source: [] },
        evaluatedTraces: [],
        otherEvaluatedTraces: [],
      });

      jest.mocked(useSetInitialTimeFilter).mockReturnValue({
        isInitialTimeFilterLoading: false,
      });

      jest.mocked(useSearchMlflowTraces).mockReturnValue({
        data: [],
        isLoading: false,
        isFetching: false,
        error: null,
      } as any);

      renderComponent();
      await waitForRoutesToBeRendered();

      // Test error in column selector
      const columnSelectorButton = screen.getByTestId('column-selector-button');
      await userEvent.click(columnSelectorButton);

      await waitFor(() => {
        // Look for the error message in the dropdown
        const dropdowns = screen.getAllByText('Fetching traces failed');
        // Should have exactly 2: one in the table and one in the column selector dropdown
        expect(dropdowns.length).toBe(2);
      });

      // Close column selector by clicking outside
      await userEvent.click(document.body);

      // Test error in sort dropdown
      const sortButton = screen.getByTestId('sort-select-dropdown');
      await userEvent.click(sortButton);

      await waitFor(() => {
        const dropdowns = screen.getAllByText('Fetching traces failed');
        // Should have exactly 2: one in the table and one in the sort dropdown
        expect(dropdowns.length).toBe(2);
      });

      // Close sort dropdown by clicking outside
      await userEvent.click(document.body);

      // Test error in filter dropdown
      const filterButton = screen.getByRole('button', { name: /filter/i });
      await userEvent.click(filterButton);

      await waitFor(() => {
        const dropdowns = screen.getAllByText('Fetching traces failed');
        // Should have exactly 2: one in the table and one in the filter dropdown
        expect(dropdowns.length).toBe(2);
      });
    });
  });

  describe('Loading state combinations', () => {
    loadingStateMatrix.forEach(
      ({
        isMetadataLoading,
        isInitialTimeFilterLoading,
        traceInfosLoading,
        testName,
        uiInLoadingState,
        uiShowingData,
      }) => {
        it(`should render correctly when ${testName}`, async () => {
          // Set up mocks for this specific test case
          const mockColumns = [
            {
              id: REQUEST_TIME_COLUMN_ID,
              type: TracesTableColumnType.TRACE_INFO,
              group: TracesTableColumnGroup.INFO,
              label: 'Request Time',
            },
          ];

          jest.mocked(useMlflowTracesTableMetadata).mockReturnValue({
            assessmentInfos: [],
            allColumns: mockColumns,
            totalCount: 10,
            isLoading: isMetadataLoading,
            error: null,
            isEmpty: false,
            tableFilterOptions: { source: [] },
            evaluatedTraces: [],
            otherEvaluatedTraces: [],
          });

          jest.mocked(useSetInitialTimeFilter).mockReturnValue({
            isInitialTimeFilterLoading: isInitialTimeFilterLoading,
          });

          jest.mocked(useSearchMlflowTraces).mockReturnValue({
            data: traceInfosLoading ? undefined : [{ trace_id: 'test-trace-1' }],
            isLoading: traceInfosLoading,
            isFetching: traceInfosLoading,
            error: null,
          } as any);

          renderComponent();
          await waitForRoutesToBeRendered();

          // Verify table loading state
          if (uiInLoadingState.includes('table')) {
            // When traceInfosLoading is true, we should see ParagraphSkeleton components
            // The component renders 10 skeleton lines
            const loadingTexts = screen.getAllByText('Loading...');
            expect(loadingTexts.length).toBeGreaterThan(0);
          }

          // Verify table showing data
          if (uiShowingData.includes('table')) {
            // When table is showing data, we should not see loading skeletons
            expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
          }

          // Verify toolbar state - check if toolbar exists
          // The toolbar is always rendered, containing the search and filters
          const filterButton = screen.getByRole('button', { name: /filter/i });
          expect(filterButton).toBeInTheDocument();

          if (uiShowingData.includes('toolbar (selecting a selector shows spinner)')) {
            // When metadata is loading, clicking the column selector will show loading state in the dialog
            const columnSelectorButton = screen.getByTestId('column-selector-button');
            expect(columnSelectorButton).toBeInTheDocument();

            // Click the button to open the dialog
            await userEvent.click(columnSelectorButton);

            // Verify the dialog shows loading state
            await waitFor(() => {
              // The DialogComboboxContent with loading=true should have aria-busy
              const dialogContent = screen.getByLabelText(/columns options/i);
              expect(dialogContent).toHaveAttribute('aria-busy', 'true');
            });

            // Test sort dropdown shows spinner when loading
            const sortButton = screen.getByTestId('sort-select-dropdown');
            expect(sortButton).toBeInTheDocument();

            await userEvent.click(sortButton);

            await waitFor(() => {
              // Sort dropdown should show loading spinner
              expect(screen.getByTestId('sort-dropdown-loading')).toBeInTheDocument();
            });

            // Test filter dropdown shows spinner when loading
            await userEvent.click(filterButton);

            await waitFor(() => {
              // Filter dropdown should show loading spinner
              expect(screen.getByTestId('filter-dropdown-loading')).toBeInTheDocument();
            });
          } else if (uiShowingData.includes('toolbar')) {
            // When metadata is not loading, column selector should work normally
            const columnSelectorButton = screen.getByTestId('column-selector-button');
            expect(columnSelectorButton).toBeInTheDocument();

            // Click the button to verify it opens the dialog with content
            await userEvent.click(columnSelectorButton);

            // Verify the dialog content is shown without loading state
            await waitFor(() => {
              const dialogContent = screen.getByLabelText(/columns options/i);
              expect(dialogContent).toHaveAttribute('aria-busy', 'false');
            });

            // Test sort dropdown shows content when not loading
            const sortButton = screen.getByTestId('sort-select-dropdown');
            expect(sortButton).toBeInTheDocument();

            await userEvent.click(sortButton);

            await waitFor(() => {
              // Sort dropdown should show search input when not loading
              // Just check that a search input exists in the dropdown content
              const dropdownContent = screen.getByRole('menu');
              const searchInput = within(dropdownContent).getByPlaceholderText(/search/i);
              expect(searchInput).toBeInTheDocument();
            });

            // Test filter dropdown shows content when not loading
            const filterButton = screen.getByRole('button', { name: /filter/i });
            expect(filterButton).toBeInTheDocument();

            await userEvent.click(filterButton);

            await waitFor(() => {
              // Filter dropdown should show filter form when not loading
              // Look for buttons that are typically in the filter form
              const applyButton = screen.getByRole('button', { name: /apply/i });
              expect(applyButton).toBeInTheDocument();
            });
          }
        });
      },
    );
  });
});
