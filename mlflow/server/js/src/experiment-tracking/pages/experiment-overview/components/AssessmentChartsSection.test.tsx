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
  INTERNAL_ASSESSMENT_ISSUE_DISCOVERY_JUDGE,
} from '@databricks/web-shared/model-trace-explorer';
import { setupServer } from '../../../../common/utils/setup-msw';
import { rest } from 'msw';
import { OverviewChartProvider } from '../OverviewChartContext';
import { MemoryRouter } from '../../../../common/utils/RoutingUtils';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';

// Helper to create a distribution data point (ASSESSMENT_COUNT with ASSESSMENT_NAME + ASSESSMENT_VALUE)
const createDistributionDataPoint = (assessmentName: string, assessmentValue: string, count: number) => ({
  metric_name: AssessmentMetricKey.ASSESSMENT_COUNT,
  dimensions: {
    [AssessmentDimensionKey.ASSESSMENT_NAME]: assessmentName,
    [AssessmentDimensionKey.ASSESSMENT_VALUE]: assessmentValue,
  },
  values: { [AggregationType.COUNT]: count },
});

// Helper to create a time-series data point (ASSESSMENT_VALUE with ASSESSMENT_NAME + time_bucket)
const createTimeSeriesDataPoint = (assessmentName: string, timeBucket: string, avgValue: number) => ({
  metric_name: AssessmentMetricKey.ASSESSMENT_VALUE,
  dimensions: {
    [AssessmentDimensionKey.ASSESSMENT_NAME]: assessmentName,
    time_bucket: timeBucket,
  },
  values: { [AggregationType.AVG]: avgValue },
});

// Helper to create a simple count data point for useHasAssessmentsOutsideTimeRange
const createSimpleCountDataPoint = (assessmentName: string, count: number) => ({
  metric_name: AssessmentMetricKey.ASSESSMENT_COUNT,
  dimensions: { [AssessmentDimensionKey.ASSESSMENT_NAME]: assessmentName },
  values: { [AggregationType.COUNT]: count },
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

  // Helper to setup MSW handler that returns different responses based on metric_name or metric_names
  // distributionData: for ASSESSMENT_COUNT query (distribution with name+value)
  // timeSeriesData: for ASSESSMENT_VALUE query (time-series with name+time_bucket)
  const setupTraceMetricsHandler = (distributionData: any[], timeSeriesData: any[] = []) => {
    server.use(
      rest.post(getAjaxUrl('ajax-api/3.0/mlflow/traces/metrics'), async (req, res, ctx) => {
        const body = await req.json();
        const metricName: string | undefined = body.metric_name;
        const metricNames: string[] = body.metric_names ?? [];
        if (
          metricName === AssessmentMetricKey.ASSESSMENT_COUNT ||
          metricNames.includes(AssessmentMetricKey.ASSESSMENT_COUNT)
        ) {
          return res(ctx.json({ data_points: distributionData }));
        }
        // ASSESSMENT_VALUE query (time-series)
        return res(ctx.json({ data_points: timeSeriesData }));
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
        rest.post(getAjaxUrl('ajax-api/3.0/mlflow/traces/metrics'), (_req, res, ctx) => {
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
        rest.post(getAjaxUrl('ajax-api/3.0/mlflow/traces/metrics'), (_req, res, ctx) => {
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

    it('should treat only internal issue discovery judge as no assessments', async () => {
      setupTraceMetricsHandler(
        [createSimpleCountDataPoint(INTERNAL_ASSESSMENT_ISSUE_DISCOVERY_JUDGE, 50)],
        [createTimeSeriesDataPoint(INTERNAL_ASSESSMENT_ISSUE_DISCOVERY_JUDGE, '2025-12-22T10:00:00Z', 1)],
      );

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('No assessments available')).toBeInTheDocument();
        expect(screen.getByText('Monitor quality metrics from scorers')).toBeInTheDocument();
      });
    });

    it('should not suggest widening time range when only hidden judge exists outside the range', async () => {
      server.use(
        rest.post(getAjaxUrl('ajax-api/3.0/mlflow/traces/metrics'), async (req, res, ctx) => {
          const body = await req.json();
          const hasNoTimeRange = !('start_time_ms' in body) || body.start_time_ms === null;
          if (hasNoTimeRange && body.metric_name === AssessmentMetricKey.ASSESSMENT_COUNT) {
            return res(
              ctx.json({ data_points: [createSimpleCountDataPoint(INTERNAL_ASSESSMENT_ISSUE_DISCOVERY_JUDGE, 10)] }),
            );
          }
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('No assessments available')).toBeInTheDocument();
        expect(screen.getByText('Monitor quality metrics from scorers')).toBeInTheDocument();
      });
      expect(screen.queryByText(/Try selecting a longer time range/)).not.toBeInTheDocument();
    });

    it('should render time range message when assessments exist outside the current time range', async () => {
      // Setup handler that returns empty for time-filtered queries but data for non-time-filtered queries
      server.use(
        rest.post(getAjaxUrl('ajax-api/3.0/mlflow/traces/metrics'), async (req, res, ctx) => {
          const body = await req.json();
          // If query has no time range (from useHasAssessmentsOutsideTimeRange), return data
          const hasNoTimeRange = !('start_time_ms' in body) || body.start_time_ms === null;
          if (hasNoTimeRange && body.metric_name === AssessmentMetricKey.ASSESSMENT_COUNT) {
            return res(ctx.json({ data_points: [createSimpleCountDataPoint('SomeAssessment', 10)] }));
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
    // Distribution data: each assessment has one numeric value for simplicity
    const mockDistributionData = [
      createDistributionDataPoint('Correctness', '0.85', 100),
      createDistributionDataPoint('Relevance', '0.72', 80),
      createDistributionDataPoint('Fluency', '0.9', 60),
    ];
    // Time-series data
    const mockTimeSeriesData = [
      createTimeSeriesDataPoint('Correctness', '2025-12-22T10:00:00Z', 0.85),
      createTimeSeriesDataPoint('Relevance', '2025-12-22T10:00:00Z', 0.72),
      createTimeSeriesDataPoint('Fluency', '2025-12-22T10:00:00Z', 0.9),
    ];

    it('should render section header with title', async () => {
      setupTraceMetricsHandler(mockDistributionData, mockTimeSeriesData);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Quality Insights')).toBeInTheDocument();
      });
    });

    it('should render section description', async () => {
      setupTraceMetricsHandler(mockDistributionData, mockTimeSeriesData);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Quality metrics computed by scorers.')).toBeInTheDocument();
      });
    });

    it('should render a chart for each assessment', async () => {
      setupTraceMetricsHandler(mockDistributionData, mockTimeSeriesData);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('assessment-chart-Correctness')).toBeInTheDocument();
        expect(screen.getByTestId('assessment-chart-Fluency')).toBeInTheDocument();
        expect(screen.getByTestId('assessment-chart-Relevance')).toBeInTheDocument();
      });
    });

    it('should display average values for numeric assessments', async () => {
      setupTraceMetricsHandler(mockDistributionData, mockTimeSeriesData);

      renderComponent();

      await waitFor(() => {
        // Weighted averages: single value per name, so avg = that value
        // Values appear in both summary table and chart headers
        expect(screen.getAllByText('0.85').length).toBeGreaterThanOrEqual(1);
        expect(screen.getAllByText('0.72').length).toBeGreaterThanOrEqual(1);
        expect(screen.getAllByText('0.90').length).toBeGreaterThanOrEqual(1);
      });
    });

    it('should sort assessments alphabetically', async () => {
      setupTraceMetricsHandler(
        [
          createDistributionDataPoint('Zebra', '0.5', 10),
          createDistributionDataPoint('Alpha', '0.8', 20),
          createDistributionDataPoint('Middle', '0.6', 15),
        ],
        [
          createTimeSeriesDataPoint('Zebra', '2025-12-22T10:00:00Z', 0.5),
          createTimeSeriesDataPoint('Alpha', '2025-12-22T10:00:00Z', 0.8),
          createTimeSeriesDataPoint('Middle', '2025-12-22T10:00:00Z', 0.6),
        ],
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
      // String assessment has non-numeric values, numeric has numeric values
      setupTraceMetricsHandler(
        [
          createDistributionDataPoint('StringAssessment', 'pass', 30),
          createDistributionDataPoint('StringAssessment', 'fail', 20),
          createDistributionDataPoint('NumericAssessment', '0.75', 30),
        ],
        [createTimeSeriesDataPoint('NumericAssessment', '2025-12-22T10:00:00Z', 0.75)],
      );

      renderComponent();

      await waitFor(() => {
        // Both assessments should be rendered
        expect(screen.getByTestId('assessment-chart-StringAssessment')).toBeInTheDocument();
        expect(screen.getByTestId('assessment-chart-NumericAssessment')).toBeInTheDocument();
      });
    });

    it('should hide internal issue discovery judge from quality charts', async () => {
      setupTraceMetricsHandler(
        [
          createDistributionDataPoint(INTERNAL_ASSESSMENT_ISSUE_DISCOVERY_JUDGE, '1', 99),
          createDistributionDataPoint('UserJudge', '1', 10),
        ],
        [createTimeSeriesDataPoint('UserJudge', '2025-12-22T10:00:00Z', 0.5)],
      );

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('assessment-chart-UserJudge')).toBeInTheDocument();
        expect(
          screen.queryByTestId(`assessment-chart-${INTERNAL_ASSESSMENT_ISSUE_DISCOVERY_JUDGE}`),
        ).not.toBeInTheDocument();
      });
    });
  });

  describe('API call parameters', () => {
    it('should call API with correct parameters for distribution (COUNT) query', async () => {
      let capturedCountRequest: any = null;

      server.use(
        rest.post(getAjaxUrl('ajax-api/3.0/mlflow/traces/metrics'), async (req, res, ctx) => {
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
          dimensions: [AssessmentDimensionKey.ASSESSMENT_NAME, AssessmentDimensionKey.ASSESSMENT_VALUE],
          filters: [`assessment.${AssessmentFilterKey.TYPE} = "${AssessmentTypeValue.FEEDBACK}"`],
        });
      });
    });

    it('should call API with correct parameters for time-series (AVG) query', async () => {
      let capturedAvgRequest: any = null;

      server.use(
        rest.post(getAjaxUrl('ajax-api/3.0/mlflow/traces/metrics'), async (req, res, ctx) => {
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
          time_interval_seconds: timeIntervalSeconds,
          filters: [`assessment.${AssessmentFilterKey.TYPE} = "${AssessmentTypeValue.FEEDBACK}"`],
        });
      });
    });

    it('should include time range in API calls', async () => {
      let capturedCountRequest: any = null;

      server.use(
        rest.post(getAjaxUrl('ajax-api/3.0/mlflow/traces/metrics'), async (req, res, ctx) => {
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
    it('should handle data points with missing assessment_name in distribution query', async () => {
      setupTraceMetricsHandler(
        [
          createDistributionDataPoint('ValidName', '0.8', 10),
          {
            metric_name: AssessmentMetricKey.ASSESSMENT_COUNT,
            dimensions: { [AssessmentDimensionKey.ASSESSMENT_VALUE]: '0.5' }, // Missing assessment_name
            values: { [AggregationType.COUNT]: 5 },
          },
        ],
        [createTimeSeriesDataPoint('ValidName', '2025-12-22T10:00:00Z', 0.8)],
      );

      renderComponent();

      await waitFor(() => {
        // Should only render the valid assessment
        expect(screen.getByTestId('assessment-chart-ValidName')).toBeInTheDocument();
        expect(screen.queryAllByTestId(/^assessment-chart-/)).toHaveLength(1);
      });
    });

    it('should render chart even when assessment is string-type (no numeric avg)', async () => {
      // String assessment has non-numeric values
      setupTraceMetricsHandler(
        [
          createDistributionDataPoint('StringAssessment', 'pass', 15),
          createDistributionDataPoint('StringAssessment', 'fail', 5),
        ],
        [], // No time-series data for string assessments
      );

      renderComponent();

      await waitFor(() => {
        // Should still render the chart (just without moving average)
        expect(screen.getByTestId('assessment-chart-StringAssessment')).toBeInTheDocument();
      });
    });
  });
});
