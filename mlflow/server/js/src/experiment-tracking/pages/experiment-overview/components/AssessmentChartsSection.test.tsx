import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { AssessmentChartsSection } from './AssessmentChartsSection';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import {
  MetricViewType,
  AggregationType,
  AssessmentMetricKey,
  AssessmentFilterKey,
  AssessmentTypeValue,
  AssessmentDimensionKey,
} from '@databricks/web-shared/model-trace-explorer';
import { setupServer } from '../../../../common/utils/setup-msw';
import { rest } from 'msw';

// Helper to create an assessment data point with name and avg value
const createAssessmentDataPoint = (assessmentName: string, avgValue: number) => ({
  metric_name: AssessmentMetricKey.ASSESSMENT_VALUE,
  dimensions: { [AssessmentDimensionKey.ASSESSMENT_NAME]: assessmentName },
  values: { [AggregationType.AVG]: avgValue },
});

describe('AssessmentChartsSection', () => {
  const testExperimentId = 'test-experiment-123';
  const startTimeMs = new Date('2025-12-22T10:00:00Z').getTime();
  const endTimeMs = new Date('2025-12-22T12:00:00Z').getTime();
  const timeIntervalSeconds = 3600;

  const timeBuckets = [
    new Date('2025-12-22T10:00:00Z').getTime(),
    new Date('2025-12-22T11:00:00Z').getTime(),
    new Date('2025-12-22T12:00:00Z').getTime(),
  ];

  const defaultProps: {
    experimentId: string;
    startTimeMs: number;
    endTimeMs: number;
    timeIntervalSeconds: number;
    timeBuckets: number[];
    searchQuery?: string;
  } = {
    experimentId: testExperimentId,
    startTimeMs,
    endTimeMs,
    timeIntervalSeconds,
    timeBuckets,
  };

  const server = setupServer();

  const createQueryClient = () =>
    new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

  const renderComponent = (props: Partial<typeof defaultProps> = {}) => {
    const queryClient = createQueryClient();
    return renderWithIntl(
      <QueryClientProvider client={queryClient}>
        <DesignSystemProvider>
          <AssessmentChartsSection {...defaultProps} {...props} />
        </DesignSystemProvider>
      </QueryClientProvider>,
    );
  };

  // Helper to setup MSW handler for the trace metrics endpoint
  const setupTraceMetricsHandler = (dataPoints: any[] | undefined) => {
    server.use(
      rest.post('ajax-api/3.0/mlflow/traces/metrics', (_req, res, ctx) => {
        return res(ctx.json({ data_points: dataPoints }));
      }),
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
    // Default: return empty data points
    setupTraceMetricsHandler([]);
  });

  describe('loading state', () => {
    it('should render loading skeleton while data is being fetched', async () => {
      // Never resolve the request to keep loading
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', (_req, res, ctx) => {
          return res(ctx.delay('infinite'));
        }),
      );

      renderComponent();

      // Check that actual chart content is not rendered during loading
      expect(screen.queryByText('Scorer Insights')).not.toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('should render error message when API call fails', async () => {
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', (_req, res, ctx) => {
          return res(ctx.status(500), ctx.json({ error_code: 'INTERNAL_ERROR', message: 'API Error' }));
        }),
      );

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Failed to load chart data')).toBeInTheDocument();
      });
    });
  });

  describe('empty state', () => {
    it('should render empty state when no assessments are available', async () => {
      setupTraceMetricsHandler([]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('No assessments available')).toBeInTheDocument();
      });
    });
  });

  describe('with data', () => {
    const mockDataPoints = [
      createAssessmentDataPoint('Correctness', 0.85),
      createAssessmentDataPoint('Relevance', 0.72),
      createAssessmentDataPoint('Fluency', 0.9),
    ];

    it('should render section header with title', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Scorer Insights')).toBeInTheDocument();
      });
    });

    it('should render section description', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Quality metrics computed by scorers.')).toBeInTheDocument();
      });
    });

    it('should render a chart for each assessment', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('assessment-chart-Correctness')).toBeInTheDocument();
        expect(screen.getByTestId('assessment-chart-Fluency')).toBeInTheDocument();
        expect(screen.getByTestId('assessment-chart-Relevance')).toBeInTheDocument();
      });
    });

    it('should display average values for each assessment', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        // Average values are displayed in the chart headers
        expect(screen.getByText('0.85')).toBeInTheDocument();
        expect(screen.getByText('0.72')).toBeInTheDocument();
        expect(screen.getByText('0.90')).toBeInTheDocument();
      });
    });

    it('should sort assessments alphabetically', async () => {
      setupTraceMetricsHandler([
        createAssessmentDataPoint('Zebra', 0.5),
        createAssessmentDataPoint('Alpha', 0.8),
        createAssessmentDataPoint('Middle', 0.6),
      ]);

      renderComponent();

      await waitFor(() => {
        const charts = screen.getAllByTestId(/^assessment-chart-/);
        expect(charts[0]).toHaveAttribute('data-testid', 'assessment-chart-Alpha');
        expect(charts[1]).toHaveAttribute('data-testid', 'assessment-chart-Middle');
        expect(charts[2]).toHaveAttribute('data-testid', 'assessment-chart-Zebra');
      });
    });
  });

  describe('API call parameters', () => {
    it('should call API with correct parameters for fetching assessments', async () => {
      let capturedRequestBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedRequestBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent();

      await waitFor(() => {
        expect(capturedRequestBody).toMatchObject({
          experiment_ids: [testExperimentId],
          view_type: MetricViewType.ASSESSMENTS,
          metric_name: AssessmentMetricKey.ASSESSMENT_VALUE,
          aggregations: [{ aggregation_type: AggregationType.AVG }],
          dimensions: [AssessmentDimensionKey.ASSESSMENT_NAME],
          filters: [`assessment.${AssessmentFilterKey.TYPE} = "${AssessmentTypeValue.FEEDBACK}"`],
        });
      });
    });

    it('should include time range in API call', async () => {
      let capturedRequestBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedRequestBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent();

      await waitFor(() => {
        expect(capturedRequestBody.start_time_ms).toBe(startTimeMs);
        expect(capturedRequestBody.end_time_ms).toBe(endTimeMs);
      });
    });
  });

  describe('data extraction', () => {
    it('should handle data points with missing assessment_name', async () => {
      setupTraceMetricsHandler([
        createAssessmentDataPoint('ValidName', 0.8),
        {
          metric_name: AssessmentMetricKey.ASSESSMENT_VALUE,
          dimensions: {}, // Missing assessment_name
          values: { [AggregationType.AVG]: 0.5 },
        },
      ]);

      renderComponent();

      await waitFor(() => {
        // Should only render the valid assessment
        expect(screen.getByTestId('assessment-chart-ValidName')).toBeInTheDocument();
        expect(screen.queryAllByTestId(/^assessment-chart-/)).toHaveLength(1);
      });
    });

    it('should handle data points with missing avg value', async () => {
      setupTraceMetricsHandler([
        {
          metric_name: AssessmentMetricKey.ASSESSMENT_VALUE,
          dimensions: { [AssessmentDimensionKey.ASSESSMENT_NAME]: 'NoAvgValue' },
          values: {}, // Missing AVG value
        },
      ]);

      renderComponent();

      await waitFor(() => {
        // Should not render any charts
        expect(screen.queryAllByTestId(/^assessment-chart-/)).toHaveLength(0);
      });
    });
  });

  describe('searchQuery filtering', () => {
    it('should filter assessments by searchQuery', async () => {
      setupTraceMetricsHandler([
        createAssessmentDataPoint('Correctness', 0.85),
        createAssessmentDataPoint('Relevance', 0.72),
        createAssessmentDataPoint('Fluency', 0.9),
      ]);

      renderComponent({ searchQuery: 'Correct' });

      await waitFor(() => {
        expect(screen.getByTestId('assessment-chart-Correctness')).toBeInTheDocument();
      });

      expect(screen.queryByTestId('assessment-chart-Relevance')).not.toBeInTheDocument();
      expect(screen.queryByTestId('assessment-chart-Fluency')).not.toBeInTheDocument();
    });

    it('should filter assessments case-insensitively', async () => {
      setupTraceMetricsHandler([
        createAssessmentDataPoint('Correctness', 0.85),
        createAssessmentDataPoint('Relevance', 0.72),
      ]);

      renderComponent({ searchQuery: 'correctness' });

      await waitFor(() => {
        expect(screen.getByTestId('assessment-chart-Correctness')).toBeInTheDocument();
      });

      expect(screen.queryByTestId('assessment-chart-Relevance')).not.toBeInTheDocument();
    });

    it('should show all assessments when searchQuery is empty', async () => {
      setupTraceMetricsHandler([
        createAssessmentDataPoint('Correctness', 0.85),
        createAssessmentDataPoint('Relevance', 0.72),
      ]);

      renderComponent({ searchQuery: '' });

      await waitFor(() => {
        expect(screen.getByTestId('assessment-chart-Correctness')).toBeInTheDocument();
        expect(screen.getByTestId('assessment-chart-Relevance')).toBeInTheDocument();
      });
    });

    it('should show all assessments when searchQuery is whitespace only', async () => {
      setupTraceMetricsHandler([
        createAssessmentDataPoint('Correctness', 0.85),
        createAssessmentDataPoint('Relevance', 0.72),
      ]);

      renderComponent({ searchQuery: '   ' });

      await waitFor(() => {
        expect(screen.getByTestId('assessment-chart-Correctness')).toBeInTheDocument();
        expect(screen.getByTestId('assessment-chart-Relevance')).toBeInTheDocument();
      });
    });

    it('should show empty state when searchQuery matches no assessments', async () => {
      setupTraceMetricsHandler([
        createAssessmentDataPoint('Correctness', 0.85),
        createAssessmentDataPoint('Relevance', 0.72),
      ]);

      renderComponent({ searchQuery: 'nonexistent' });

      await waitFor(() => {
        expect(screen.getByText('No assessments available')).toBeInTheDocument();
      });
    });

    it('should support partial matching in searchQuery', async () => {
      setupTraceMetricsHandler([
        createAssessmentDataPoint('Correctness', 0.85),
        createAssessmentDataPoint('Relevance', 0.72),
        createAssessmentDataPoint('Fluency', 0.9),
      ]);

      renderComponent({ searchQuery: 'ness' });

      await waitFor(() => {
        expect(screen.getByTestId('assessment-chart-Correctness')).toBeInTheDocument();
      });

      expect(screen.queryByTestId('assessment-chart-Relevance')).not.toBeInTheDocument();
      expect(screen.queryByTestId('assessment-chart-Fluency')).not.toBeInTheDocument();
    });
  });
});
