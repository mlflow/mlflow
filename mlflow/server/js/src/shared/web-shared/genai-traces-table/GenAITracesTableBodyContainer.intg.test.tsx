import { render, screen, waitFor } from '@testing-library/react';
import type { ComponentProps } from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { getUser } from '@databricks/web-shared/global-settings';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';

import { GenAITracesTableBodyContainer } from './GenAITracesTableBodyContainer';
// eslint-disable-next-line import/no-namespace
import * as GenAiTracesTableUtils from './GenAiTracesTable.utils';
import { createTestTraceInfoV3, createTestAssessmentInfo, createTestColumns } from './index';
import type { TraceInfoV3 } from './types';
import { TestRouter, testRoute } from './utils/RoutingTestUtils';

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

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(30000);

// Mock necessary modules
jest.mock('@databricks/web-shared/global-settings', () => ({
  getUser: jest.fn(),
}));

jest.mock('@databricks/web-shared/hooks', () => {
  return {
    ...jest.requireActual<typeof import('@databricks/web-shared/hooks')>('@databricks/web-shared/hooks'),
    getLocalStorageItemByParams: jest.fn().mockReturnValue({ hiddenColumns: undefined }),
    useLocalStorage: jest.fn().mockReturnValue([{}, jest.fn()]),
  };
});

const testExperimentId = 'test-experiment-id';
const testRunUuid = 'test-run-uuid';
const testCompareToRunUuid = 'compare-run-uuid';

describe('GenAITracesTableBodyContainer - integration test', () => {
  beforeEach(() => {
    // Mock user ID
    jest.mocked(getUser).mockImplementation(() => 'test.user@mlflow.org');

    // Mocked returned timestamp
    jest.spyOn(Date, 'now').mockImplementation(() => 1000000);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  const waitForViewToBeReady = () =>
    waitFor(() => {
      expect(screen.getByText(/Request/)).toBeInTheDocument();
    });

  const renderTestComponent = (
    currentTraceInfoV3: TraceInfoV3[],
    compareToTraceInfoV3: TraceInfoV3[] = [],
    additionalProps: Partial<ComponentProps<typeof GenAITracesTableBodyContainer>> = {},
  ) => {
    const defaultAssessmentInfos = [
      createTestAssessmentInfo('overall_assessment', 'Overall Assessment', 'pass-fail'),
      createTestAssessmentInfo('quality_score', 'Quality Score', 'numeric'),
      createTestAssessmentInfo('is_correct', 'Is Correct', 'boolean'),
    ];

    const defaultColumns = createTestColumns(defaultAssessmentInfos);

    const defaultProps: ComponentProps<typeof GenAITracesTableBodyContainer> = {
      experimentId: testExperimentId,
      currentRunDisplayName: 'Test Run',
      runUuid: testRunUuid,
      compareToRunUuid: testCompareToRunUuid,
      compareToRunDisplayName: 'Compare Run',
      assessmentInfos: defaultAssessmentInfos,
      currentTraceInfoV3,
      compareToTraceInfoV3,
      selectedColumns: defaultColumns,
      allColumns: defaultColumns,
      tableSort: undefined,
      filters: [],
      setFilters: jest.fn(),
      getTrace: jest.fn().mockResolvedValue(undefined),
      getRunColor: jest.fn().mockReturnValue('#000000'),
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
                  <GenAITracesTableBodyContainer {...defaultProps} />
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

  it('renders table with single trace', async () => {
    const traceInfo = createTestTraceInfoV3(
      'trace-1',
      'request-1',
      'Hello world',
      [
        { name: 'overall_assessment', value: 'yes', dtype: 'pass-fail' },
        { name: 'quality_score', value: 0.85, dtype: 'numeric' },
      ],
      testExperimentId,
    );

    renderTestComponent([traceInfo]);

    await waitForViewToBeReady();

    // Verify basic table structure is rendered
    expect(screen.getByText('Hello world')).toBeInTheDocument();
    expect(screen.getByText('trace-1')).toBeInTheDocument();
  });

  it('renders table with multiple traces', async () => {
    const traceInfos = [
      createTestTraceInfoV3(
        'trace-1',
        'request-1',
        'Hello world 1',
        [{ name: 'overall_assessment', value: 'yes', dtype: 'pass-fail' }],
        testExperimentId,
      ),
      createTestTraceInfoV3(
        'trace-2',
        'request-2',
        'Hello world 2',
        [{ name: 'overall_assessment', value: 'no', dtype: 'pass-fail' }],
        testExperimentId,
      ),
    ];

    renderTestComponent(traceInfos);

    await waitForViewToBeReady();

    // Verify both traces are rendered
    expect(screen.getByText('Hello world 1')).toBeInTheDocument();
    expect(screen.getByText('Hello world 2')).toBeInTheDocument();
    expect(screen.getByText('trace-1')).toBeInTheDocument();
    expect(screen.getByText('trace-2')).toBeInTheDocument();
  });

  it('renders table with comparison data', async () => {
    const currentTraceInfos = [
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

    const compareToTraceInfos = [
      createTestTraceInfoV3(
        'trace-2',
        'request-2',
        'Hello world compare',
        [
          { name: 'overall_assessment', value: 'no', dtype: 'pass-fail' },
          { name: 'quality_score', value: 0.75, dtype: 'numeric' },
        ],
        testExperimentId,
      ),
    ];

    renderTestComponent(currentTraceInfos, compareToTraceInfos);

    await waitForViewToBeReady();

    // Verify both current and comparison data are rendered
    expect(screen.getByText('Hello world')).toBeInTheDocument();
    expect(screen.getByText('Hello world compare')).toBeInTheDocument();
  });

  it('handles different assessment data types', async () => {
    const traceInfo = createTestTraceInfoV3(
      'trace-1',
      'request-1',
      'Hello world',
      [
        { name: 'overall_assessment', value: 'yes', dtype: 'pass-fail' },
        { name: 'quality_score', value: 0.85, dtype: 'numeric' },
        { name: 'is_correct', value: true, dtype: 'boolean' },
        { name: 'description', value: 'Good response', dtype: 'string' },
      ],
      testExperimentId,
    );

    const assessmentInfos = [
      createTestAssessmentInfo('overall_assessment', 'Overall Assessment', 'pass-fail'),
      createTestAssessmentInfo('quality_score', 'Quality Score', 'numeric'),
      createTestAssessmentInfo('is_correct', 'Is Correct', 'boolean'),
      createTestAssessmentInfo('description', 'Description', 'string'),
    ];

    const columns = createTestColumns(assessmentInfos);

    renderTestComponent([traceInfo], [], {
      assessmentInfos,
      selectedColumns: columns,
      allColumns: columns,
    });

    await waitForViewToBeReady();

    // Verify different assessment types are rendered
    expect(screen.getByText('Hello world')).toBeInTheDocument();
    expect(screen.getByText('trace-1')).toBeInTheDocument();

    // Verify assessment data is present with correct display values
    expect(screen.getByText('Pass')).toBeInTheDocument(); // overall_assessment value (yes -> Pass)
    expect(screen.getByText('0.85')).toBeInTheDocument(); // quality_score value
    expect(screen.getByText('True')).toBeInTheDocument(); // is_correct value (true -> True)
    expect(screen.getByText('Good response')).toBeInTheDocument(); // description value
  });

  it('handles empty trace data', async () => {
    renderTestComponent([]);

    await waitForViewToBeReady();

    // Verify empty state is handled gracefully
    expect(screen.getByText(/No traces found/)).toBeInTheDocument();
  });

  it('handles trace with no assessments', async () => {
    const traceInfo = createTestTraceInfoV3('trace-1', 'request-1', 'Hello world', [], testExperimentId);

    renderTestComponent([traceInfo]);

    await waitForViewToBeReady();

    // Verify trace is rendered even without assessments
    expect(screen.getByText('Hello world')).toBeInTheDocument();
    expect(screen.getByText('trace-1')).toBeInTheDocument();
  });

  it('handles selected columns filtering', async () => {
    const traceInfo = createTestTraceInfoV3(
      'trace-1',
      'request-1',
      'Hello world',
      [
        { name: 'overall_assessment', value: 'yes', dtype: 'pass-fail' },
        { name: 'quality_score', value: 0.85, dtype: 'numeric' },
      ],
      testExperimentId,
    );

    const assessmentInfos = [
      createTestAssessmentInfo('overall_assessment', 'Overall Assessment', 'pass-fail'),
      createTestAssessmentInfo('quality_score', 'Quality Score', 'numeric'),
    ];

    // Only select the overall assessment column
    const allColumns = createTestColumns(assessmentInfos);
    const selectedColumns = allColumns.filter(
      (col) => col.id === 'request' || col.id === 'assessment_overall_assessment',
    );

    renderTestComponent([traceInfo], [], {
      assessmentInfos,
      selectedColumns,
      allColumns,
    });

    await waitForViewToBeReady();

    // Verify only selected columns are rendered
    expect(screen.getByText('Hello world')).toBeInTheDocument();
    expect(screen.queryByText('trace-1')).not.toBeInTheDocument();
    expect(screen.getByText('Overall Assessment')).toBeInTheDocument();
    expect(screen.queryByText('Quality Score')).not.toBeInTheDocument();
  });

  it('handles assessment aggregates computation', async () => {
    const traceInfos = [
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

    const assessmentInfos = [
      createTestAssessmentInfo('overall_assessment', 'Overall Assessment', 'pass-fail'),
      createTestAssessmentInfo('quality_score', 'Quality Score', 'numeric'),
    ];

    const columns = createTestColumns(assessmentInfos);

    renderTestComponent(traceInfos, [], {
      assessmentInfos,
      selectedColumns: columns,
      allColumns: columns,
    });

    await waitForViewToBeReady();

    // Verify both traces are rendered with their assessments
    expect(screen.getByText('Hello world 1')).toBeInTheDocument();
    expect(screen.getByText('Hello world 2')).toBeInTheDocument();

    // Check for correct aggregate values
    // For pass-fail, expect both "Pass" and "Fail" to be present
    expect(screen.getByText('Pass')).toBeInTheDocument();
    expect(screen.getByText('Fail')).toBeInTheDocument();

    // For numeric, check for the average (0.8 or 0.80)
    expect(screen.getByText(/0\.8/)).toBeInTheDocument();
  });

  it('handles undefined optional props gracefully', async () => {
    const traceInfo = createTestTraceInfoV3('trace-1', 'request-1', 'Hello world', [], testExperimentId);

    renderTestComponent([traceInfo], [], {
      currentRunDisplayName: undefined,
      runUuid: undefined,
      compareToRunUuid: undefined,
      compareToRunDisplayName: undefined,
      getRunColor: undefined,
      onTraceTagsEdit: undefined,
    });

    await waitForViewToBeReady();

    // Verify component renders without errors
    expect(screen.getByText('Hello world')).toBeInTheDocument();
    expect(screen.getByText('trace-1')).toBeInTheDocument();
  });
});
