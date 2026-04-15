import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { TraceErrorsChart } from './TraceErrorsChart';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { AggregationType, TraceMetricKey, TraceDimensionKey } from '@databricks/web-shared/model-trace-explorer';
import { setupServer } from '../../../../common/utils/setup-msw';
import { rest } from 'msw';
import { OverviewChartProvider } from '../OverviewChartContext';
import { MemoryRouter } from '../../../../common/utils/RoutingUtils';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';

// Helper to create a data point with trace_status dimension
const createStatusDataPoint = (timeBucket: string, count: number, status: string) => ({
  metric_name: TraceMetricKey.TRACE_COUNT,
  dimensions: { time_bucket: timeBucket, [TraceDimensionKey.TRACE_STATUS]: status },
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

  const renderComponent = (contextOverrides: Partial<typeof defaultContextProps> = {}) => {
    const queryClient = createQueryClient();
    const contextProps = { ...defaultContextProps, ...contextOverrides };
    return renderWithIntl(
      <MemoryRouter>
        <QueryClientProvider client={queryClient}>
          <DesignSystemProvider>
            <OverviewChartProvider {...contextProps}>
              <TraceErrorsChart />
            </OverviewChartProvider>
          </DesignSystemProvider>
        </QueryClientProvider>
      </MemoryRouter>,
    );
  };

  // Helper to setup MSW handler — returns status-dimensioned data points
  const setupTraceMetricsHandler = (dataPoints: any[]) => {
    server.use(
      rest.post(getAjaxUrl('ajax-api/3.0/mlflow/traces/metrics'), (_req, res, ctx) => {
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
        rest.post(getAjaxUrl('ajax-api/3.0/mlflow/traces/metrics'), (_req, res, ctx) => {
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
    // Status-dimensioned data: OK and ERROR rows per time bucket
    const mockDataPoints = [
      createStatusDataPoint('2025-12-22T10:00:00Z', 95, 'OK'),
      createStatusDataPoint('2025-12-22T10:00:00Z', 5, 'ERROR'),
      createStatusDataPoint('2025-12-22T11:00:00Z', 190, 'OK'),
      createStatusDataPoint('2025-12-22T11:00:00Z', 10, 'ERROR'),
    ];

    it('should render chart with all time buckets', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('composed-chart')).toBeInTheDocument();
      });

      // Should have all 3 time buckets (10:00, 11:00, 12:00)
      expect(screen.getByTestId('composed-chart')).toHaveAttribute('data-count', '3');
    });

    it('should display the "Errors" title', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Errors')).toBeInTheDocument();
      });
    });

    it('should display the total error count', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      // Total errors should be 5 + 10 = 15
      await waitFor(() => {
        expect(screen.getByText(/15/)).toBeInTheDocument();
      });
    });

    it('should display the overall error rate', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      // Error rate: (5 + 10) / (95+5 + 190+10) = 15/300 = 5%
      await waitFor(() => {
        expect(screen.getByText(/Overall error rate: 5\.0%/)).toBeInTheDocument();
      });
    });

    it('should render both bar and line series', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('bar-Error Count')).toBeInTheDocument();
        expect(screen.getByTestId('line-Error Rate')).toBeInTheDocument();
      });
    });

    it('should render reference line with AVG label', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        const referenceLine = screen.getByTestId('reference-line');
        expect(referenceLine).toBeInTheDocument();
        expect(referenceLine).toHaveAttribute('data-label', expect.stringContaining('AVG'));
      });
    });

    it('should fill missing time buckets with zeros', async () => {
      // Only provide data for one time bucket
      setupTraceMetricsHandler([
        createStatusDataPoint('2025-12-22T10:00:00Z', 95, 'OK'),
        createStatusDataPoint('2025-12-22T10:00:00Z', 5, 'ERROR'),
      ]);

      renderComponent();

      // Chart should still show all 3 time buckets
      await waitFor(() => {
        expect(screen.getByTestId('composed-chart')).toHaveAttribute('data-count', '3');
      });
    });
  });

  describe('API call parameters', () => {
    it('should call API with trace_status dimension (no separate error filter)', async () => {
      let capturedRequest: any = null;

      server.use(
        rest.post(getAjaxUrl('ajax-api/3.0/mlflow/traces/metrics'), async (req, res, ctx) => {
          capturedRequest = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent();

      await waitFor(() => {
        expect(capturedRequest).not.toBeNull();
        expect(capturedRequest?.dimensions).toEqual([TraceDimensionKey.TRACE_STATUS]);
        // No ERROR filter — status is extracted client-side from the dimension
        expect(capturedRequest?.filters).toBeUndefined();
      });
    });

    it('should use provided time interval', async () => {
      let capturedRequest: any = null;

      server.use(
        rest.post(getAjaxUrl('ajax-api/3.0/mlflow/traces/metrics'), async (req, res, ctx) => {
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
      setupTraceMetricsHandler([
        {
          metric_name: TraceMetricKey.TRACE_COUNT,
          dimensions: { time_bucket: '2025-12-22T10:00:00Z', [TraceDimensionKey.TRACE_STATUS]: 'ERROR' },
          values: {}, // Missing COUNT value - will be treated as 0
        },
        createStatusDataPoint('2025-12-22T11:00:00Z', 10, 'ERROR'),
        createStatusDataPoint('2025-12-22T10:00:00Z', 100, 'OK'),
        createStatusDataPoint('2025-12-22T11:00:00Z', 200, 'OK'),
      ]);

      renderComponent();

      // Should still render with all time buckets and show total errors of 10 (0 + 10)
      await waitFor(() => {
        expect(screen.getByTestId('composed-chart')).toHaveAttribute('data-count', '3');
      });
      expect(screen.getByText(/10/)).toBeInTheDocument();
    });

    it('should handle zero total count gracefully (no division by zero)', async () => {
      setupTraceMetricsHandler([
        createStatusDataPoint('2025-12-22T10:00:00Z', 0, 'OK'),
        createStatusDataPoint('2025-12-22T10:00:00Z', 0, 'ERROR'),
      ]);

      renderComponent();

      // Should render with all time buckets, error rate should be 0%
      await waitFor(() => {
        expect(screen.getByTestId('composed-chart')).toHaveAttribute('data-count', '3');
      });
      expect(screen.getByText(/Overall error rate: 0\.0%/)).toBeInTheDocument();
    });

    it('should handle data points with missing time_bucket', async () => {
      setupTraceMetricsHandler([
        {
          metric_name: TraceMetricKey.TRACE_COUNT,
          dimensions: { [TraceDimensionKey.TRACE_STATUS]: 'ERROR' }, // Missing time_bucket
          values: { [AggregationType.COUNT]: 5 },
        },
        {
          metric_name: TraceMetricKey.TRACE_COUNT,
          dimensions: { [TraceDimensionKey.TRACE_STATUS]: 'OK' }, // Missing time_bucket
          values: { [AggregationType.COUNT]: 100 },
        },
      ]);

      renderComponent();

      // Should still render the chart with all generated time buckets (all with 0 values in chart)
      await waitFor(() => {
        expect(screen.getByTestId('composed-chart')).toHaveAttribute('data-count', '3');
      });
    });
  });
});
