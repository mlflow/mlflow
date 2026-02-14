import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { ToolLatencyChart } from './ToolLatencyChart';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { AggregationType, SpanMetricKey, SpanDimensionKey } from '@databricks/web-shared/model-trace-explorer';
import { setupServer } from '../../../../common/utils/setup-msw';
import { rest } from 'msw';
import { OverviewChartProvider } from '../OverviewChartContext';

// Helper to create a tool latency data point
const createToolLatencyDataPoint = (timeBucket: string, toolName: string, avgLatency: number) => ({
  metric_name: SpanMetricKey.LATENCY,
  dimensions: {
    time_bucket: timeBucket,
    [SpanDimensionKey.SPAN_NAME]: toolName,
  },
  values: { [AggregationType.AVG]: avgLatency },
});

describe('ToolLatencyChart', () => {
  const testExperimentId = 'test-experiment-123';
  const startTimeMs = new Date('2025-12-22T10:00:00Z').getTime();
  const endTimeMs = new Date('2025-12-22T12:00:00Z').getTime();
  const timeIntervalSeconds = 3600; // 1 hour

  const timeBuckets = [
    new Date('2025-12-22T10:00:00Z').getTime(),
    new Date('2025-12-22T11:00:00Z').getTime(),
    new Date('2025-12-22T12:00:00Z').getTime(),
  ];

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
      <QueryClientProvider client={queryClient}>
        <DesignSystemProvider>
          <OverviewChartProvider {...contextProps}>
            <ToolLatencyChart />
          </OverviewChartProvider>
        </DesignSystemProvider>
      </QueryClientProvider>,
    );
  };

  // Helper to setup MSW handler for the trace metrics endpoint
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
      expect(screen.queryByText('Latency Comparison')).not.toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('should render error message when API call fails', async () => {
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', (_req, res, ctx) => {
          return res(ctx.status(500), ctx.json({ error: 'API Error' }));
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
      // Default handler returns empty array
      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('No data available for the selected time range')).toBeInTheDocument();
      });
    });

    it('should render empty state when time range is not provided', async () => {
      // Default handler returns empty array
      renderComponent({ startTimeMs: undefined, endTimeMs: undefined, timeBuckets: [] });

      await waitFor(() => {
        expect(screen.getByText('No data available for the selected time range')).toBeInTheDocument();
      });
    });
  });

  describe('with data', () => {
    const mockDataPoints = [
      createToolLatencyDataPoint('2025-12-22T10:00:00Z', 'delivery_estimate', 150),
      createToolLatencyDataPoint('2025-12-22T11:00:00Z', 'delivery_estimate', 180),
      createToolLatencyDataPoint('2025-12-22T12:00:00Z', 'delivery_estimate', 200),
      createToolLatencyDataPoint('2025-12-22T10:00:00Z', 'get_order_status', 120),
      createToolLatencyDataPoint('2025-12-22T11:00:00Z', 'get_order_status', 140),
    ];

    it('should render chart when data is available', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });

      expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', '3');
    });

    it('should display the chart title', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Latency Comparison')).toBeInTheDocument();
      });
    });
  });

  describe('with single tool', () => {
    it('should render chart with single tool data', async () => {
      setupTraceMetricsHandler([
        createToolLatencyDataPoint('2025-12-22T10:00:00Z', 'single_tool', 100),
        createToolLatencyDataPoint('2025-12-22T11:00:00Z', 'single_tool', 150),
      ]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });

      expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', '3');
    });
  });
});
