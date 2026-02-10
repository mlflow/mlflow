import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor, fireEvent } from '@testing-library/react';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { TraceRequestsChart } from './TraceRequestsChart';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MetricViewType, AggregationType, TraceMetricKey } from '@databricks/web-shared/model-trace-explorer';
import { setupServer } from '../../../../common/utils/setup-msw';
import { rest } from 'msw';
import { OverviewChartProvider } from '../OverviewChartContext';

// Helper to create a single data point
const createTraceCountDataPoint = (timeBucket: string, count: number) => ({
  metric_name: TraceMetricKey.TRACE_COUNT,
  dimensions: { time_bucket: timeBucket },
  values: { [AggregationType.COUNT]: count },
});

describe('TraceRequestsChart', () => {
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
            <TraceRequestsChart />
          </OverviewChartProvider>
        </DesignSystemProvider>
      </QueryClientProvider>,
    );
  };

  // Helper to setup MSW handler for trace metrics endpoint
  const setupTraceMetricsHandler = (dataPoints: any[]) => {
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
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', (_req, res, ctx) => {
          return res(ctx.delay('infinite'));
        }),
      );

      renderComponent();

      // Check that actual chart content is not rendered during loading
      expect(screen.queryByText('Traces')).not.toBeInTheDocument();
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
      setupTraceMetricsHandler([]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('No data available for the selected time range')).toBeInTheDocument();
      });
    });

    it('should render empty state when time range is not provided', async () => {
      setupTraceMetricsHandler([]);

      renderComponent({ startTimeMs: undefined, endTimeMs: undefined, timeBuckets: [] });

      await waitFor(() => {
        expect(screen.getByText('No data available for the selected time range')).toBeInTheDocument();
      });
    });
  });

  describe('with data', () => {
    const mockDataPoints = [
      createTraceCountDataPoint('2025-12-22T10:00:00Z', 42),
      createTraceCountDataPoint('2025-12-22T11:00:00Z', 58),
      createTraceCountDataPoint('2025-12-22T12:00:00Z', 100),
    ];

    it('should render chart with all time buckets', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      });

      // Verify the bar chart has all 3 time buckets
      expect(screen.getByTestId('bar-chart')).toHaveAttribute('data-count', '3');
    });

    it('should display the total request count', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      // Total should be 42 + 58 + 100 = 200
      await waitFor(() => {
        expect(screen.getByText('200')).toBeInTheDocument();
      });
    });

    it('should display the "Traces" title', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Traces')).toBeInTheDocument();
      });
    });

    it('should display average reference line', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      // Average should be (42 + 58 + 100) / 3 = 66.67, rounded to 67
      await waitFor(() => {
        const referenceLine = screen.getByTestId('reference-line');
        expect(referenceLine).toBeInTheDocument();
        expect(referenceLine).toHaveAttribute('data-label', 'AVG (67)');
      });
    });

    it('should format large numbers with locale formatting', async () => {
      setupTraceMetricsHandler([createTraceCountDataPoint('2025-12-22T10:00:00Z', 1234567)]);

      renderComponent();

      await waitFor(() => {
        // Check for locale-formatted number (1,234,567 in US locale)
        expect(screen.getByText('1,234,567')).toBeInTheDocument();
      });
    });

    it('should fill missing time buckets with zeros', async () => {
      // Only provide data for one time bucket
      setupTraceMetricsHandler([createTraceCountDataPoint('2025-12-22T10:00:00Z', 100)]);

      renderComponent();

      // Chart should still show all 3 time buckets
      await waitFor(() => {
        expect(screen.getByTestId('bar-chart')).toHaveAttribute('data-count', '3');
      });

      // Total should only count the one data point
      expect(screen.getByText('100')).toBeInTheDocument();
    });
  });

  describe('API call parameters', () => {
    it('should call API with correct parameters', async () => {
      let capturedRequest: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedRequest = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent();

      await waitFor(() => {
        expect(capturedRequest).toMatchObject({
          experiment_ids: [testExperimentId],
          view_type: MetricViewType.TRACES,
          metric_name: TraceMetricKey.TRACE_COUNT,
          aggregations: [{ aggregation_type: AggregationType.COUNT }],
        });
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

    it('should use provided time interval for hourly grouping', async () => {
      let capturedRequest: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedRequest = await req.json();
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
          capturedRequest = await req.json();
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
    it('should handle data points with missing values gracefully', async () => {
      setupTraceMetricsHandler([
        {
          metric_name: TraceMetricKey.TRACE_COUNT,
          dimensions: { time_bucket: '2025-12-22T10:00:00Z' },
          values: {}, // Missing COUNT value - will be treated as 0
        },
        createTraceCountDataPoint('2025-12-22T11:00:00Z', 50),
      ]);

      renderComponent();

      // Should still render with all time buckets and show total of 50 (0 + 50)
      await waitFor(() => {
        expect(screen.getByTestId('bar-chart')).toHaveAttribute('data-count', '3');
      });
      expect(screen.getByText('50')).toBeInTheDocument();
    });

    it('should handle data points with missing time_bucket', async () => {
      setupTraceMetricsHandler([
        {
          metric_name: TraceMetricKey.TRACE_COUNT,
          dimensions: {}, // Missing time_bucket - won't be mapped to any bucket
          values: { [AggregationType.COUNT]: 25 },
        },
      ]);

      renderComponent();

      // Should still render with all time buckets (all with 0 values in chart, but total still counts)
      await waitFor(() => {
        expect(screen.getByTestId('bar-chart')).toHaveAttribute('data-count', '3');
      });
      // Total count still includes the data point
      expect(screen.getByText('25')).toBeInTheDocument();
    });
  });

  describe('zoom functionality', () => {
    const mockDataPoints = [
      createTraceCountDataPoint('2025-12-22T10:00:00Z', 42),
      createTraceCountDataPoint('2025-12-22T11:00:00Z', 58),
      createTraceCountDataPoint('2025-12-22T12:00:00Z', 100),
    ];

    it('should not show zoom out button initially', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Traces')).toBeInTheDocument();
      });

      // Zoom Out button should not be visible when not zoomed
      expect(screen.queryByText('Zoom Out')).not.toBeInTheDocument();
    });

    it('should render chart with correct data count initially', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      });

      // Initial state: all 3 data points visible
      expect(screen.getByTestId('bar-chart')).toHaveAttribute('data-count', '3');
    });

    it('should display initial average based on all data points', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      // Average should be (42 + 58 + 100) / 3 = 66.67, rounded to 67
      await waitFor(() => {
        const referenceLine = screen.getByTestId('reference-line');
        expect(referenceLine).toBeInTheDocument();
        expect(referenceLine).toHaveAttribute('data-label', 'AVG (67)');
      });
    });

    it('should have mouse event handlers attached to chart', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      });

      const chart = screen.getByTestId('bar-chart');

      // Verify mouse events can be fired without errors (handlers are wired up)
      expect(() => {
        fireEvent.mouseDown(chart);
        fireEvent.mouseMove(chart);
        fireEvent.mouseUp(chart);
      }).not.toThrow();
    });
  });
});
