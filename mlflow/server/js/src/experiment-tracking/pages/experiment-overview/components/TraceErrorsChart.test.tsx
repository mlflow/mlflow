import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { TraceErrorsChart } from './TraceErrorsChart';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { AggregationType, TraceMetricKey } from '@databricks/web-shared/model-trace-explorer';
import { setupServer } from '../../../../common/utils/setup-msw';
import { rest } from 'msw';
import { OverviewChartProvider } from '../OverviewChartContext';

// Helper to create an error count data point
const createErrorCountDataPoint = (timeBucket: string, count: number) => ({
  metric_name: TraceMetricKey.TRACE_COUNT,
  dimensions: { time_bucket: timeBucket },
  values: { [AggregationType.COUNT]: count },
});

// Helper to create a total count data point
const createTotalCountDataPoint = (timeBucket: string, count: number) => ({
  metric_name: TraceMetricKey.TRACE_COUNT,
  dimensions: { time_bucket: timeBucket },
  values: { [AggregationType.COUNT]: count },
});

describe('TraceErrorsChart', () => {
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
            <TraceErrorsChart />
          </OverviewChartProvider>
        </DesignSystemProvider>
      </QueryClientProvider>,
    );
  };

  // Helper to setup MSW handler for trace metrics endpoint with routing based on filters
  const setupTraceMetricsHandler = (errorDataPoints: any[], totalDataPoints: any[]) => {
    server.use(
      rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
        const body = await req.json();
        // Check if this is an error count request (has error filter) or total count request
        const hasErrorFilter = body.filters?.some((f: string) => f.includes('ERROR'));
        if (hasErrorFilter) {
          return res(ctx.json({ data_points: errorDataPoints }));
        }
        return res(ctx.json({ data_points: totalDataPoints }));
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
      expect(screen.queryByText('Errors')).not.toBeInTheDocument();
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
    const mockErrorDataPoints = [
      createErrorCountDataPoint('2025-12-22T10:00:00Z', 5),
      createErrorCountDataPoint('2025-12-22T11:00:00Z', 10),
    ];

    const mockTotalDataPoints = [
      createTotalCountDataPoint('2025-12-22T10:00:00Z', 100),
      createTotalCountDataPoint('2025-12-22T11:00:00Z', 200),
    ];

    it('should render chart with all time buckets', async () => {
      setupTraceMetricsHandler(mockErrorDataPoints, mockTotalDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('composed-chart')).toBeInTheDocument();
      });

      // Should have all 3 time buckets (10:00, 11:00, 12:00)
      expect(screen.getByTestId('composed-chart')).toHaveAttribute('data-count', '3');
    });

    it('should display the "Errors" title', async () => {
      setupTraceMetricsHandler(mockErrorDataPoints, mockTotalDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Errors')).toBeInTheDocument();
      });
    });

    it('should display the total error count', async () => {
      setupTraceMetricsHandler(mockErrorDataPoints, mockTotalDataPoints);

      renderComponent();

      // Total errors should be 5 + 10 = 15
      await waitFor(() => {
        expect(screen.getByText(/15/)).toBeInTheDocument();
      });
    });

    it('should display the overall error rate', async () => {
      setupTraceMetricsHandler(mockErrorDataPoints, mockTotalDataPoints);

      renderComponent();

      // Error rate: (5 + 10) / (100 + 200) = 15/300 = 5%
      await waitFor(() => {
        expect(screen.getByText(/Overall error rate: 5\.0%/)).toBeInTheDocument();
      });
    });

    it('should render both bar and line series', async () => {
      setupTraceMetricsHandler(mockErrorDataPoints, mockTotalDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('bar-Error Count')).toBeInTheDocument();
        expect(screen.getByTestId('line-Error Rate')).toBeInTheDocument();
      });
    });

    it('should render reference line with AVG label', async () => {
      setupTraceMetricsHandler(mockErrorDataPoints, mockTotalDataPoints);

      renderComponent();

      await waitFor(() => {
        const referenceLine = screen.getByTestId('reference-line');
        expect(referenceLine).toBeInTheDocument();
        expect(referenceLine).toHaveAttribute('data-label', expect.stringContaining('AVG'));
      });
    });

    it('should fill missing time buckets with zeros', async () => {
      // Only provide data for one time bucket
      setupTraceMetricsHandler(
        [createErrorCountDataPoint('2025-12-22T10:00:00Z', 5)],
        [createTotalCountDataPoint('2025-12-22T10:00:00Z', 100)],
      );

      renderComponent();

      // Chart should still show all 3 time buckets
      await waitFor(() => {
        expect(screen.getByTestId('composed-chart')).toHaveAttribute('data-count', '3');
      });
    });
  });

  describe('API call parameters', () => {
    it('should call API with error filter for error count request', async () => {
      let capturedErrorRequest: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          const body = await req.json();
          const hasErrorFilter = body.filters?.some((f: string) => f.includes('ERROR'));
          if (hasErrorFilter) {
            capturedErrorRequest = body;
          }
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent();

      await waitFor(() => {
        expect(capturedErrorRequest?.filters).toContain('trace.status = "ERROR"');
      });
    });

    it('should call API without filter for total count request', async () => {
      let capturedTotalRequest: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          const body = await req.json();
          const hasErrorFilter = body.filters?.some((f: string) => f.includes('ERROR'));
          if (!hasErrorFilter) {
            capturedTotalRequest = body;
          }
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent();

      await waitFor(() => {
        expect(capturedTotalRequest).not.toBeNull();
        expect(capturedTotalRequest?.filters).toBeUndefined();
      });
    });

    it('should use provided time interval', async () => {
      let capturedRequest: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedRequest = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent({ timeIntervalSeconds: 60 });

      await waitFor(() => {
        expect(capturedRequest?.time_interval_seconds).toBe(60);
      });
    });
  });

  describe('data transformation', () => {
    it('should handle data points with missing values gracefully', async () => {
      setupTraceMetricsHandler(
        [
          {
            metric_name: TraceMetricKey.TRACE_COUNT,
            dimensions: { time_bucket: '2025-12-22T10:00:00Z' },
            values: {}, // Missing COUNT value - will be treated as 0
          },
          createErrorCountDataPoint('2025-12-22T11:00:00Z', 10),
        ],
        [
          createTotalCountDataPoint('2025-12-22T10:00:00Z', 100),
          createTotalCountDataPoint('2025-12-22T11:00:00Z', 200),
        ],
      );

      renderComponent();

      // Should still render with all time buckets and show total of 10 (0 + 10)
      await waitFor(() => {
        expect(screen.getByTestId('composed-chart')).toHaveAttribute('data-count', '3');
      });
      expect(screen.getByText(/10/)).toBeInTheDocument();
    });

    it('should handle zero total count gracefully (no division by zero)', async () => {
      setupTraceMetricsHandler(
        [createErrorCountDataPoint('2025-12-22T10:00:00Z', 0)],
        [createTotalCountDataPoint('2025-12-22T10:00:00Z', 0)],
      );

      renderComponent();

      // Should render with all time buckets, error rate should be 0%
      await waitFor(() => {
        expect(screen.getByTestId('composed-chart')).toHaveAttribute('data-count', '3');
      });
      expect(screen.getByText(/Overall error rate: 0\.0%/)).toBeInTheDocument();
    });

    it('should handle data points with missing time_bucket', async () => {
      setupTraceMetricsHandler(
        [
          {
            metric_name: TraceMetricKey.TRACE_COUNT,
            dimensions: {}, // Missing time_bucket - won't be mapped to any bucket
            values: { [AggregationType.COUNT]: 5 },
          },
        ],
        [
          {
            metric_name: TraceMetricKey.TRACE_COUNT,
            dimensions: {}, // Missing time_bucket
            values: { [AggregationType.COUNT]: 100 },
          },
        ],
      );

      renderComponent();

      // Should still render the chart with all generated time buckets (all with 0 values in chart)
      await waitFor(() => {
        expect(screen.getByTestId('composed-chart')).toHaveAttribute('data-count', '3');
      });
    });
  });
});
