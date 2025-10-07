import type { ComponentProps } from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { DesignSystemProvider } from '@databricks/design-system';
import { ApolloProvider, ApolloClient, InMemoryCache } from '@mlflow/mlflow/src/common/utils/graphQLHooks';

import { TracesV3Logs } from './TracesV3Logs';
import {
  createTestTraceInfoV3,
  createTestAssessmentInfo,
  createTestColumns,
  useMlflowTracesTableMetadata,
  useSearchMlflowTraces,
  useSelectedColumns,
  convertTraceInfoV3ToRunEvalEntry,
} from '@databricks/web-shared/genai-traces-table';

import { getUser } from '@databricks/web-shared/global-settings';
import type { NetworkRequestError } from '@databricks/web-shared/errors';
import { TestRouter, testRoute, waitForRoutesToBeRendered } from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';

// Mock the virtualizer to render all rows in tests
jest.mock('@tanstack/react-virtual', () => {
  const actual = jest.requireActual<typeof import('@tanstack/react-virtual')>('@tanstack/react-virtual');
  return {
    ...actual,
    useVirtualizer: (opts: any) => {
      return {
        getVirtualItems: () =>
          Array.from({ length: opts.count }, (_, i) => ({
            index: i,
            key: i,
            start: i * 120,
            size: 120,
            measureElement: () => {},
          })),
        getTotalSize: () => opts.count * 120,
        measureElement: () => {},
      };
    },
  };
});

// Mock necessary modules
jest.mock('@databricks/web-shared/global-settings', () => ({
  getUser: jest.fn(),
}));

jest.mock('@databricks/web-shared/hooks', () => {
  return {
    getLocalStorageItemByParams: jest.fn().mockReturnValue({ hiddenColumns: undefined }),
    useLocalStorage: jest.fn().mockReturnValue([{}, jest.fn()]),
  };
});

// Mock the genai-traces-table hooks
jest.mock('@databricks/web-shared/genai-traces-table', () => {
  const actual = jest.requireActual<typeof import('@databricks/web-shared/genai-traces-table')>(
    '@databricks/web-shared/genai-traces-table',
  );
  return {
    ...actual,
    useSearchMlflowTraces: jest.fn().mockReturnValue({ data: undefined, isLoading: false, error: null }),
    useMlflowTracesTableMetadata: jest.fn().mockReturnValue({
      assessmentInfos: [],
      allColumns: [],
      totalCount: 0,
      isLoading: true,
      error: null,
      isEmpty: false,
    }),
    useSelectedColumns: jest
      .fn()
      .mockReturnValue({ selectedColumns: [], toggleColumns: jest.fn(), setSelectedColumns: jest.fn() }),
    useFilters: jest.fn().mockReturnValue([[], jest.fn()]),
    useTableSort: jest.fn().mockReturnValue([undefined, jest.fn()]),
    getEvalTabTotalTracesLimit: jest.fn().mockReturnValue(100),
    invalidateMlflowSearchTracesCache: jest.fn().mockResolvedValue(undefined),
    getTracesTagKeys: jest.fn().mockReturnValue([]),
  };
});

// Mock MLflow service
jest.mock('@mlflow/mlflow/src/experiment-tracking/sdk/MlflowService', () => ({
  MlflowService: {
    getExperimentTraceInfoV3: jest.fn(),
    getExperimentTraceData: jest.fn(),
  },
}));

// Mock hooks
jest.mock('../../../evaluations/hooks/useDeleteTraces', () => ({
  useDeleteTracesMutation: jest.fn().mockReturnValue({ mutateAsync: jest.fn().mockResolvedValue(undefined) }),
}));

jest.mock('../../../traces/hooks/useEditExperimentTraceTags', () => ({
  useEditExperimentTraceTags: jest.fn().mockReturnValue({ showEditTagsModalForTrace: jest.fn(), EditTagsModal: null }),
}));

jest.mock('./hooks/useSetInitialTimeFilter', () => ({
  useSetInitialTimeFilter: jest.fn().mockReturnValue({
    isInitialTimeFilterLoading: false,
  }),
}));

jest.mock('@mlflow/mlflow/src/common/utils/MarkdownUtils', () => ({
  useMarkdownConverter: jest.fn().mockReturnValue(jest.fn()),
}));

const testExperimentId = 'test-experiment-id';
const testEndpointName = 'test-endpoint';

describe('TracesV3Logs - integration test', () => {
  beforeEach(() => {
    // Mock user ID
    jest.mocked(getUser).mockImplementation(() => 'test.user@mlflow.org');

    // Mocked returned timestamp
    jest.spyOn(Date, 'now').mockImplementation(() => 1000000);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  const renderTestComponent = (additionalProps: Partial<ComponentProps<typeof TracesV3Logs>> = {}) => {
    const defaultProps: ComponentProps<typeof TracesV3Logs> = {
      experimentId: testExperimentId,
      endpointName: testEndpointName,
      timeRange: {
        startTime: '2024-01-01T00:00:00Z',
        endTime: '2024-01-31T23:59:59Z',
      },
      loggedModelId: 'test-model-id',
      ...additionalProps,
    };

    const TestComponent = () => {
      return (
        <TestRouter
          routes={[
            testRoute(
              <DesignSystemProvider>
                <QueryClientProvider
                  client={
                    new QueryClient({
                      logger: {
                        error: () => {},
                        log: () => {},
                        warn: () => {},
                      },
                    })
                  }
                >
                  <TracesV3Logs {...defaultProps} />
                </QueryClientProvider>
              </DesignSystemProvider>,
            ),
          ]}
        />
      );
    };

    return render(
      <IntlProvider locale="en">
        <TestComponent />
      </IntlProvider>,
    );
  };

  it('renders loading state when metadata is loading', async () => {
    renderTestComponent();
    await waitForRoutesToBeRendered();

    expect(screen.getAllByText('Loading...')).toHaveLength(10);
  });

  it('renders error state when metadata fails to load', async () => {
    jest.mocked(useMlflowTracesTableMetadata).mockReturnValue({
      assessmentInfos: [],
      allColumns: [],
      totalCount: 0,
      isLoading: false,
      error: new Error('Failed to fetch metadata') as unknown as NetworkRequestError,
      isEmpty: false,
      tableFilterOptions: { source: [] },
      evaluatedTraces: [],
      otherEvaluatedTraces: [],
    });
    renderTestComponent();
    await waitForRoutesToBeRendered();

    expect(screen.getByText('Fetching traces failed')).toBeInTheDocument();
    expect(screen.getByText('Failed to fetch metadata')).toBeInTheDocument();
  });

  it('renders empty state when no traces exist', async () => {
    jest.mocked(useMlflowTracesTableMetadata).mockReturnValue({
      assessmentInfos: [],
      allColumns: [],
      totalCount: 0,
      isLoading: false,
      error: null,
      isEmpty: true,
      tableFilterOptions: { source: [] },
      evaluatedTraces: [],
      otherEvaluatedTraces: [],
    });

    // Wrap in ApolloProvider for this test
    const mockApolloClient = new ApolloClient({ uri: '/graphql', cache: new InMemoryCache() });
    const defaultProps = {
      experimentId: testExperimentId,
      endpointName: testEndpointName,
      timeRange: {
        startTime: '2024-01-01T00:00:00Z',
        endTime: '2024-01-31T23:59:59Z',
      },
      loggedModelId: 'test-model-id',
    };
    const TestComponent = () => (
      <TestRouter
        routes={[
          testRoute(
            <ApolloProvider client={mockApolloClient}>
              <DesignSystemProvider>
                <QueryClientProvider
                  client={new QueryClient({ logger: { error: () => {}, log: () => {}, warn: () => {} } })}
                >
                  <TracesV3Logs {...defaultProps} />
                </QueryClientProvider>
              </DesignSystemProvider>
            </ApolloProvider>,
          ),
        ]}
      />
    );
    render(
      <IntlProvider locale="en">
        <TestComponent />
      </IntlProvider>,
    );

    await waitForRoutesToBeRendered();
    // Wait for loading skeletons to disappear
    await waitFor(() => {
      expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
    });
    // Now check for the empty state text
    expect(await screen.findByText(/No traces recorded/i)).toBeInTheDocument();
  });

  it('renders traces table when traces exist', async () => {
    const mockTraceInfos = [
      createTestTraceInfoV3(
        'trace-1',
        'request-1',
        'Hello world',
        [
          { name: 'overall_assessment', value: 'yes', dtype: 'pass-fail' },
          { name: 'quality_score', value: 0.85, dtype: 'numeric' },
        ],
        testExperimentId,
      ),
    ];
    const mockAssessmentInfos = [
      createTestAssessmentInfo('overall_assessment', 'Overall Assessment', 'pass-fail'),
      createTestAssessmentInfo('quality_score', 'Quality Score', 'numeric'),
    ];
    const mockColumns = createTestColumns(mockAssessmentInfos);
    const mockTableFilterOptions = { source: [] };
    jest.mocked(useMlflowTracesTableMetadata).mockReturnValue({
      assessmentInfos: mockAssessmentInfos,
      allColumns: mockColumns,
      totalCount: 1,
      isLoading: false,
      error: null,
      isEmpty: false,
      tableFilterOptions: mockTableFilterOptions,
      evaluatedTraces: mockTraceInfos.map((trace) => convertTraceInfoV3ToRunEvalEntry(trace)),
      otherEvaluatedTraces: [],
    });
    jest.mocked(useSearchMlflowTraces).mockReturnValue({
      data: mockTraceInfos,
      isLoading: false,
      isFetching: false,
      error: null as unknown as NetworkRequestError,
    });
    jest
      .mocked(useSelectedColumns)
      .mockReturnValue({ selectedColumns: mockColumns, toggleColumns: jest.fn(), setSelectedColumns: jest.fn() });

    renderTestComponent();
    await waitForRoutesToBeRendered();

    // Wait for the table to load and verify actual trace data is rendered
    await waitFor(() => {
      expect(screen.getByText('Hello world')).toBeInTheDocument();
    });

    // Verify trace ID is displayed
    expect(screen.getByText('trace-1')).toBeInTheDocument();

    // Verify assessment data is displayed
    expect(screen.getByText('Pass')).toBeInTheDocument(); // overall_assessment value (yes -> Pass)
    expect(screen.getByText('0.85')).toBeInTheDocument(); // quality_score value

    // Verify search input is also present
    expect(screen.getByPlaceholderText('Search traces by request')).toBeInTheDocument();
  });

  it('handles traces error state', async () => {
    const mockAssessmentInfos = [createTestAssessmentInfo('overall_assessment', 'Overall Assessment', 'pass-fail')];
    const mockColumns = createTestColumns(mockAssessmentInfos);
    const mockTableFilterOptions = { source: [] };
    jest.mocked(useMlflowTracesTableMetadata).mockReturnValue({
      assessmentInfos: mockAssessmentInfos,
      allColumns: mockColumns,
      totalCount: 1,
      isLoading: false,
      error: null,
      isEmpty: false,
      tableFilterOptions: mockTableFilterOptions,
      evaluatedTraces: [],
      otherEvaluatedTraces: [],
    });
    jest.mocked(useSearchMlflowTraces).mockReturnValue({
      data: undefined,
      isLoading: false,
      isFetching: false,
      error: new Error('Failed to fetch traces') as unknown as NetworkRequestError,
    });
    jest
      .mocked(useSelectedColumns)
      .mockReturnValue({ selectedColumns: mockColumns, toggleColumns: jest.fn(), setSelectedColumns: jest.fn() });
    renderTestComponent();
    await waitForRoutesToBeRendered();
    expect(screen.getByText('Failed to fetch traces')).toBeInTheDocument();
  });

  it('handles multiple traces with assessments', async () => {
    const mockTraceInfos = [
      createTestTraceInfoV3(
        'trace-1',
        'request-1',
        'Hello world 1',
        [
          { name: 'overall_assessment', value: 'yes', dtype: 'pass-fail' },
          { name: 'quality_score', value: 0.85, dtype: 'numeric' },
        ],
        testExperimentId,
      ),
      createTestTraceInfoV3(
        'trace-2',
        'request-2',
        'Hello world 2',
        [
          { name: 'overall_assessment', value: 'no', dtype: 'pass-fail' },
          { name: 'quality_score', value: 0.75, dtype: 'numeric' },
        ],
        testExperimentId,
      ),
    ];

    const mockAssessmentInfos = [
      createTestAssessmentInfo('overall_assessment', 'Overall Assessment', 'pass-fail'),
      createTestAssessmentInfo('quality_score', 'Quality Score', 'numeric'),
    ];

    const mockColumns = createTestColumns(mockAssessmentInfos);
    const mockTableFilterOptions = { source: [] };

    jest.mocked(useMlflowTracesTableMetadata).mockReturnValue({
      assessmentInfos: mockAssessmentInfos,
      allColumns: mockColumns,
      totalCount: 2,
      isLoading: false,
      error: null,
      isEmpty: false,
      tableFilterOptions: mockTableFilterOptions,
      evaluatedTraces: mockTraceInfos.map((trace) => convertTraceInfoV3ToRunEvalEntry(trace)),
      otherEvaluatedTraces: [],
    });

    jest.mocked(useSearchMlflowTraces).mockReturnValue({
      data: mockTraceInfos,
      isLoading: false,
      isFetching: false,
      error: null as unknown as NetworkRequestError,
    });

    jest.mocked(useSelectedColumns).mockReturnValue({
      selectedColumns: mockColumns,
      toggleColumns: jest.fn(),
      setSelectedColumns: jest.fn(),
    });

    renderTestComponent();
    await waitForRoutesToBeRendered();

    // Wait for the table to load and verify both traces are rendered
    await waitFor(() => {
      expect(screen.getByText('Hello world 1')).toBeInTheDocument();
      expect(screen.getByText('Hello world 2')).toBeInTheDocument();
    });

    // Verify trace IDs are displayed
    expect(screen.getByText('trace-1')).toBeInTheDocument();
    expect(screen.getByText('trace-2')).toBeInTheDocument();

    // Verify assessment data is displayed for both traces
    expect(screen.getByText('Pass')).toBeInTheDocument(); // trace-1 overall_assessment (yes -> Pass)
    expect(screen.getByText('Fail')).toBeInTheDocument(); // trace-2 overall_assessment (no -> Fail)
    expect(screen.getByText('0.85')).toBeInTheDocument(); // trace-1 quality_score
    expect(screen.getByText('0.75')).toBeInTheDocument(); // trace-2 quality_score

    // Verify search input is also present
    expect(screen.getByPlaceholderText('Search traces by request')).toBeInTheDocument();
  });
});
