import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { TraceTokenStatsChart } from './TraceTokenStatsChart';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import {
  MetricViewType,
  AggregationType,
  TraceMetricKey,
  P50,
  P90,
  P99,
  getPercentileKey,
} from '@databricks/web-shared/model-trace-explorer';
import { setupServer } from '../../../../common/utils/setup-msw';
import { rest } from 'msw';
import { OverviewChartProvider } from '../OverviewChartContext';

// Helper to create a token stats percentile data point
const createTokenStatsDataPoint = (timeBucket: string, p50: number, p90: number, p99: number) => ({
  metric_name: TraceMetricKey.TOTAL_TOKENS,
  dimensions: { time_bucket: timeBucket },
  values: {
    [getPercentileKey(P50)]: p50,
    [getPercentileKey(P90)]: p90,
    [getPercentileKey(P99)]: p99,
  },
});

// Helper to create an AVG token stats data point
const createAvgTokenStatsDataPoint = (avg: number) => ({
  metric_name: TraceMetricKey.TOTAL_TOKENS,
  dimensions: {},
  values: { [AggregationType.AVG]: avg },
});

describe('TraceTokenStatsChart', () => {
  const testExperimentId = 'test-experiment-123';
  // Use fixed timestamps for predictable bucket generation
  const startTimeMs = new Date('2025-12-22T10:00:00Z').getTime();
  const endTimeMs = new Date('2025-12-22T12:00:00Z').getTime(); // 2 hours = 3 buckets with 1hr interval
  const timeIntervalSeconds = 3600; // 1 hour

  // Pre-computed time buckets for the test range
  const timeBuckets = [
    new Date('2025-12-22T10:00:00Z').getTime(),
    new Date('2025-12-22T11:00:00Z').getTime(),
    new Date('2025-12-22T12:00:00Z').getTime(),
  ];

  // Context props reused across tests
  const defaultContextProps = {
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

  const renderComponent = (contextOverrides: Partial<typeof defaultContextProps> = {}) => {
    const queryClient = createQueryClient();
    const contextProps = { ...defaultContextProps, ...contextOverrides };
    return renderWithIntl(
      <QueryClientProvider client={queryClient}>
        <DesignSystemProvider>
          <OverviewChartProvider {...contextProps}>
            <TraceTokenStatsChart />
          </OverviewChartProvider>
        </DesignSystemProvider>
      </QueryClientProvider>,
    );
  };

  // Helper to setup MSW handler for trace metrics endpoint with routing based on aggregations
  const setupTraceMetricsHandler = (percentileDataPoints: any[], avgDataPoints: any[]) => {
    server.use(
      rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
        const body = await req.json();
        // Check if this is a percentile request or AVG request
        const hasPercentileAggregation = body.aggregations?.some(
          (a: any) => a.aggregation_type === AggregationType.PERCENTILE,
        );
        if (hasPercentileAggregation) {
          return res(ctx.json({ data_points: percentileDataPoints }));
        }
        return res(ctx.json({ data_points: avgDataPoints }));
      }),
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
    // Default: return empty data points
    setupTraceMetricsHandler([], []);
  });

  describe('loading state', () => {
    it('should render loading skeleton while data is being fetched', async () => {
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', (_req, res, ctx) => {
          return res(ctx.delay('infinite'));
        }),
      );

      renderComponent();

      // Check that actual chart content is not rendered during loading
      expect(screen.queryByText('Tokens per Trace')).not.toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('should render error message when time series API call fails', async () => {
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

  describe('empty data state', () => {
    it('should render empty state when no data points are returned', async () => {
      setupTraceMetricsHandler([], []);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('No data available for the selected time range')).toBeInTheDocument();
      });
    });

    it('should render empty state when time range is not provided', async () => {
      setupTraceMetricsHandler([], []);

      renderComponent({ startTimeMs: undefined, endTimeMs: undefined, timeBuckets: [] });

      await waitFor(() => {
        expect(screen.getByText('No data available for the selected time range')).toBeInTheDocument();
      });
    });
  });

  describe('with data', () => {
    const mockPercentileDataPoints = [
      createTokenStatsDataPoint('2025-12-22T10:00:00Z', 1500, 3500, 8000),
      createTokenStatsDataPoint('2025-12-22T11:00:00Z', 1800, 4000, 9500),
    ];

    const mockAvgDataPoints = [createAvgTokenStatsDataPoint(2500)];

    it('should render chart with all time buckets', async () => {
      setupTraceMetricsHandler(mockPercentileDataPoints, mockAvgDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });

      // Verify the line chart has all 3 time buckets (10:00, 11:00, 12:00)
      expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', '3');
    });

    it('should display all three percentile lines', async () => {
      setupTraceMetricsHandler(mockPercentileDataPoints, mockAvgDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('line-p50')).toBeInTheDocument();
        expect(screen.getByTestId('line-p90')).toBeInTheDocument();
        expect(screen.getByTestId('line-p99')).toBeInTheDocument();
      });
    });

    it('should display the average tokens in header', async () => {
      setupTraceMetricsHandler(mockPercentileDataPoints, mockAvgDataPoints);

      renderComponent();

      // Average is 2500 -> 2.50K
      await waitFor(() => {
        expect(screen.getByText('2.50K')).toBeInTheDocument();
      });
    });

    it('should display the "Tokens per Trace" title', async () => {
      setupTraceMetricsHandler(mockPercentileDataPoints, mockAvgDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Tokens per Trace')).toBeInTheDocument();
      });
    });

    it('should display "avg per trace" subtitle', async () => {
      setupTraceMetricsHandler(mockPercentileDataPoints, mockAvgDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('avg per trace')).toBeInTheDocument();
      });
    });

    it('should format token count in millions for values >= 1,000,000', async () => {
      setupTraceMetricsHandler(mockPercentileDataPoints, [createAvgTokenStatsDataPoint(1500000)]);

      renderComponent();

      // 1500000 should be displayed as 1.50M
      await waitFor(() => {
        expect(screen.getByText('1.50M')).toBeInTheDocument();
      });
    });

    it('should format token count in thousands for values >= 1,000', async () => {
      setupTraceMetricsHandler(mockPercentileDataPoints, [createAvgTokenStatsDataPoint(5000)]);

      renderComponent();

      // 5000 should be displayed as 5.00K
      await waitFor(() => {
        expect(screen.getByText('5.00K')).toBeInTheDocument();
      });
    });

    it('should render reference line with AVG label', async () => {
      setupTraceMetricsHandler(mockPercentileDataPoints, mockAvgDataPoints);

      renderComponent();

      await waitFor(() => {
        const referenceLine = screen.getByTestId('reference-line');
        expect(referenceLine).toBeInTheDocument();
        expect(referenceLine).toHaveAttribute('data-label', 'AVG (2.50K)');
      });
    });

    it('should fill missing time buckets with zeros', async () => {
      // Only provide data for one time bucket
      setupTraceMetricsHandler(
        [createTokenStatsDataPoint('2025-12-22T10:00:00Z', 1500, 3500, 8000)],
        mockAvgDataPoints,
      );

      renderComponent();

      // Chart should still show all 3 time buckets
      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', '3');
      });
    });
  });

  describe('API call parameters', () => {
    it('should call API for percentiles with correct parameters', async () => {
      let capturedPercentileRequest: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          const body = await req.json();
          const hasPercentileAggregation = body.aggregations?.some(
            (a: any) => a.aggregation_type === AggregationType.PERCENTILE,
          );
          if (hasPercentileAggregation) {
            capturedPercentileRequest = body;
          }
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent();

      await waitFor(() => {
        expect(capturedPercentileRequest).toMatchObject({
          experiment_ids: [testExperimentId],
          view_type: MetricViewType.TRACES,
          metric_name: TraceMetricKey.TOTAL_TOKENS,
          aggregations: [
            { aggregation_type: AggregationType.PERCENTILE, percentile_value: P50 },
            { aggregation_type: AggregationType.PERCENTILE, percentile_value: P90 },
            { aggregation_type: AggregationType.PERCENTILE, percentile_value: P99 },
          ],
        });
      });
    });

    it('should call API for AVG with correct parameters', async () => {
      let capturedAvgRequest: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          const body = await req.json();
          const hasAvgAggregation = body.aggregations?.some((a: any) => a.aggregation_type === AggregationType.AVG);
          if (hasAvgAggregation) {
            capturedAvgRequest = body;
          }
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent();

      await waitFor(() => {
        expect(capturedAvgRequest).toMatchObject({
          experiment_ids: [testExperimentId],
          view_type: MetricViewType.TRACES,
          metric_name: TraceMetricKey.TOTAL_TOKENS,
          aggregations: [{ aggregation_type: AggregationType.AVG }],
        });
      });
    });

    it('should use provided time interval', async () => {
      let capturedRequest: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          const body = await req.json();
          // Capture request with time_interval_seconds (percentile request)
          if (body.time_interval_seconds !== undefined) {
            capturedRequest = body;
          }
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent({ timeIntervalSeconds: 60 });

      await waitFor(() => {
        expect(capturedRequest?.time_interval_seconds).toBe(60);
      });
    });

    it('should use provided time interval for hourly grouping', async () => {
      let capturedRequest: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          const body = await req.json();
          // Capture request with time_interval_seconds (percentile request)
          if (body.time_interval_seconds !== undefined) {
            capturedRequest = body;
          }
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent({ timeIntervalSeconds: 3600 });

      await waitFor(() => {
        expect(capturedRequest?.time_interval_seconds).toBe(3600);
      });
    });

    it('should use provided time interval for daily grouping', async () => {
      let capturedRequest: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          const body = await req.json();
          // Capture request with time_interval_seconds (percentile request)
          if (body.time_interval_seconds !== undefined) {
            capturedRequest = body;
          }
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent({ timeIntervalSeconds: 86400 });

      await waitFor(() => {
        expect(capturedRequest?.time_interval_seconds).toBe(86400);
      });
    });
  });

  describe('data transformation', () => {
    it('should handle data points with missing percentile values gracefully', async () => {
      setupTraceMetricsHandler(
        [
          {
            metric_name: TraceMetricKey.TOTAL_TOKENS,
            dimensions: { time_bucket: '2025-12-22T10:00:00Z' },
            values: {}, // Missing percentile values - will be treated as 0
          },
        ],
        [createAvgTokenStatsDataPoint(100)],
      );

      renderComponent();

      // Should still render with all time buckets
      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', '3');
      });
    });

    it('should handle missing AVG data gracefully', async () => {
      setupTraceMetricsHandler([createTokenStatsDataPoint('2025-12-22T10:00:00Z', 1500, 3500, 8000)], []);

      renderComponent();

      // Should still render the chart with all time buckets
      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', '3');
      });

      // Should NOT display avg value or reference line when not available
      expect(screen.queryByTestId('reference-line')).not.toBeInTheDocument();
    });

    it('should handle data points with missing time_bucket', async () => {
      setupTraceMetricsHandler(
        [
          {
            metric_name: TraceMetricKey.TOTAL_TOKENS,
            dimensions: {}, // Missing time_bucket - won't be mapped to any bucket
            values: {
              [getPercentileKey(P50)]: 1000,
              [getPercentileKey(P90)]: 2000,
              [getPercentileKey(P99)]: 3000,
            },
          },
        ],
        [createAvgTokenStatsDataPoint(1500)],
      );

      renderComponent();

      // Should still render the chart with all generated time buckets (all with 0 values)
      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', '3');
      });
    });
  });
});
