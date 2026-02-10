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
import { OverviewChartProvider } from '../OverviewChartContext';
import { MemoryRouter } from '../../../../common/utils/RoutingUtils';

// Helper to create an assessment count data point (for getting all assessment names)
const createCountDataPoint = (assessmentName: string, count: number) => ({
  metric_name: AssessmentMetricKey.ASSESSMENT_COUNT,
  dimensions: { [AssessmentDimensionKey.ASSESSMENT_NAME]: assessmentName },
  values: { [AggregationType.COUNT]: count },
});

// Helper to create an assessment avg data point (for numeric assessments)
const createAvgDataPoint = (assessmentName: string, avgValue: number) => ({
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

  const contextProps = {
    experimentIds: [testExperimentId],
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

  const renderComponent = () => {
    const queryClient = createQueryClient();
    return renderWithIntl(
      <MemoryRouter>
        <QueryClientProvider client={queryClient}>
          <DesignSystemProvider>
            <OverviewChartProvider {...contextProps}>
              <AssessmentChartsSection />
            </OverviewChartProvider>
          </DesignSystemProvider>
        </QueryClientProvider>
      </MemoryRouter>,
    );
  };

  // Helper to setup MSW handler that returns different responses based on metric_name
  // countData: for ASSESSMENT_COUNT query (gets all assessment names)
  // avgData: for ASSESSMENT_VALUE query (gets avg for numeric assessments)
  const setupTraceMetricsHandler = (countData: any[], avgData: any[] = countData) => {
    server.use(
      rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
        const body = await req.json();
        if (body.metric_name === AssessmentMetricKey.ASSESSMENT_COUNT) {
          return res(ctx.json({ data_points: countData }));
        }
        // ASSESSMENT_VALUE query
        return res(ctx.json({ data_points: avgData }));
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
      expect(screen.queryByText('Quality Insights')).not.toBeInTheDocument();
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
    it('should render empty state with guidance when no assessments are available at all', async () => {
      setupTraceMetricsHandler([]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('No assessments available')).toBeInTheDocument();
        expect(screen.getByText('Monitor quality metrics from scorers')).toBeInTheDocument();
        expect(screen.getByText('Learn more')).toBeInTheDocument();
      });
    });

    it('should render time range message when assessments exist outside the current time range', async () => {
      // Setup handler that returns empty for time-filtered queries but data for non-time-filtered queries
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          const body = await req.json();
          // If query has no time range (from useHasAssessmentsOutsideTimeRange), return data
          // Note: undefined values are omitted when serialized to JSON, so we check if the property doesn't exist
          const hasNoTimeRange = !('start_time_ms' in body) || body.start_time_ms === null;
          if (hasNoTimeRange && body.metric_name === AssessmentMetricKey.ASSESSMENT_COUNT) {
            return res(ctx.json({ data_points: [createCountDataPoint('SomeAssessment', 10)] }));
          }
          // Time-filtered queries return empty
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent();

      await waitFor(
        () => {
          expect(screen.getByText('No assessments available')).toBeInTheDocument();
          expect(screen.getByText(/Try selecting a longer time range/)).toBeInTheDocument();
        },
        { timeout: 10000 },
      );

      // Should NOT show the full guidance
      expect(screen.queryByText('Monitor quality metrics from scorers')).not.toBeInTheDocument();
    }, 15000);
  });

  describe('with data', () => {
    const mockCountData = [
      createCountDataPoint('Correctness', 100),
      createCountDataPoint('Relevance', 80),
      createCountDataPoint('Fluency', 60),
    ];
    const mockAvgData = [
      createAvgDataPoint('Correctness', 0.85),
      createAvgDataPoint('Relevance', 0.72),
      createAvgDataPoint('Fluency', 0.9),
    ];

    it('should render section header with title', async () => {
      setupTraceMetricsHandler(mockCountData, mockAvgData);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Quality Insights')).toBeInTheDocument();
      });
    });

    it('should render section description', async () => {
      setupTraceMetricsHandler(mockCountData, mockAvgData);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Quality metrics computed by scorers.')).toBeInTheDocument();
      });
    });

    it('should render a chart for each assessment', async () => {
      setupTraceMetricsHandler(mockCountData, mockAvgData);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('assessment-chart-Correctness')).toBeInTheDocument();
        expect(screen.getByTestId('assessment-chart-Fluency')).toBeInTheDocument();
        expect(screen.getByTestId('assessment-chart-Relevance')).toBeInTheDocument();
      });
    });

    it('should display average values for numeric assessments', async () => {
      setupTraceMetricsHandler(mockCountData, mockAvgData);

      renderComponent();

      await waitFor(() => {
        // Average values are displayed in the chart headers
        expect(screen.getByText('0.85')).toBeInTheDocument();
        expect(screen.getByText('0.72')).toBeInTheDocument();
        expect(screen.getByText('0.90')).toBeInTheDocument();
      });
    });

    it('should sort assessments alphabetically', async () => {
      setupTraceMetricsHandler(
        [createCountDataPoint('Zebra', 10), createCountDataPoint('Alpha', 20), createCountDataPoint('Middle', 15)],
        [createAvgDataPoint('Zebra', 0.5), createAvgDataPoint('Alpha', 0.8), createAvgDataPoint('Middle', 0.6)],
      );

      renderComponent();

      await waitFor(() => {
        const charts = screen.getAllByTestId(/^assessment-chart-/);
        expect(charts[0]).toHaveAttribute('data-testid', 'assessment-chart-Alpha');
        expect(charts[1]).toHaveAttribute('data-testid', 'assessment-chart-Middle');
        expect(charts[2]).toHaveAttribute('data-testid', 'assessment-chart-Zebra');
      });
    });

    it('should render charts for string-type assessments without avgValue', async () => {
      // String assessment has count but no avg
      setupTraceMetricsHandler(
        [createCountDataPoint('StringAssessment', 50), createCountDataPoint('NumericAssessment', 30)],
        [createAvgDataPoint('NumericAssessment', 0.75)], // Only numeric has avg
      );

      renderComponent();

      await waitFor(() => {
        // Both assessments should be rendered
        expect(screen.getByTestId('assessment-chart-StringAssessment')).toBeInTheDocument();
        expect(screen.getByTestId('assessment-chart-NumericAssessment')).toBeInTheDocument();
      });
    });
  });

  describe('API call parameters', () => {
    it('should call API with correct parameters for COUNT query', async () => {
      let capturedCountRequest: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          const body = await req.json();
          if (body.metric_name === AssessmentMetricKey.ASSESSMENT_COUNT) {
            capturedCountRequest = body;
          }
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent();

      await waitFor(() => {
        expect(capturedCountRequest).toMatchObject({
          experiment_ids: [testExperimentId],
          view_type: MetricViewType.ASSESSMENTS,
          metric_name: AssessmentMetricKey.ASSESSMENT_COUNT,
          aggregations: [{ aggregation_type: AggregationType.COUNT }],
          dimensions: [AssessmentDimensionKey.ASSESSMENT_NAME],
          filters: [`assessment.${AssessmentFilterKey.TYPE} = "${AssessmentTypeValue.FEEDBACK}"`],
        });
      });
    });

    it('should call API with correct parameters for AVG query', async () => {
      let capturedAvgRequest: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          const body = await req.json();
          if (body.metric_name === AssessmentMetricKey.ASSESSMENT_VALUE) {
            capturedAvgRequest = body;
          }
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent();

      await waitFor(() => {
        expect(capturedAvgRequest).toMatchObject({
          experiment_ids: [testExperimentId],
          view_type: MetricViewType.ASSESSMENTS,
          metric_name: AssessmentMetricKey.ASSESSMENT_VALUE,
          aggregations: [{ aggregation_type: AggregationType.AVG }],
          dimensions: [AssessmentDimensionKey.ASSESSMENT_NAME],
          filters: [`assessment.${AssessmentFilterKey.TYPE} = "${AssessmentTypeValue.FEEDBACK}"`],
        });
      });
    });

    it('should include time range in API calls', async () => {
      let capturedCountRequest: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          const body = await req.json();
          if (body.metric_name === AssessmentMetricKey.ASSESSMENT_COUNT) {
            capturedCountRequest = body;
          }
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent();

      await waitFor(() => {
        expect(capturedCountRequest.start_time_ms).toBe(startTimeMs);
        expect(capturedCountRequest.end_time_ms).toBe(endTimeMs);
      });
    });
  });

  describe('data extraction', () => {
    it('should handle data points with missing assessment_name in count query', async () => {
      setupTraceMetricsHandler(
        [
          createCountDataPoint('ValidName', 10),
          {
            metric_name: AssessmentMetricKey.ASSESSMENT_COUNT,
            dimensions: {}, // Missing assessment_name
            values: { [AggregationType.COUNT]: 5 },
          },
        ],
        [createAvgDataPoint('ValidName', 0.8)],
      );

      renderComponent();

      await waitFor(() => {
        // Should only render the valid assessment
        expect(screen.getByTestId('assessment-chart-ValidName')).toBeInTheDocument();
        expect(screen.queryAllByTestId(/^assessment-chart-/)).toHaveLength(1);
      });
    });

    it('should render chart even when avg value is missing (string-type assessment)', async () => {
      // Assessment has count but no avg (string type)
      setupTraceMetricsHandler(
        [createCountDataPoint('StringAssessment', 20)],
        [], // No avg values
      );

      renderComponent();

      await waitFor(() => {
        // Should still render the chart (just without moving average)
        expect(screen.getByTestId('assessment-chart-StringAssessment')).toBeInTheDocument();
      });
    });
  });
});
