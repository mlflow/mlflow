import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { ToolPerformanceSummary } from './ToolPerformanceSummary';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import {
  AggregationType,
  SpanMetricKey,
  SpanDimensionKey,
  SpanStatus,
} from '@databricks/web-shared/model-trace-explorer';
import { setupServer } from '../../../../common/utils/setup-msw';
import { rest } from 'msw';

// Helper to create a count data point (grouped by tool name and status)
const createCountDataPoint = (toolName: string, status: string, count: number) => ({
  metric_name: SpanMetricKey.SPAN_COUNT,
  dimensions: {
    [SpanDimensionKey.SPAN_NAME]: toolName,
    [SpanDimensionKey.SPAN_STATUS]: status,
  },
  values: { [AggregationType.COUNT]: count },
});

// Helper to create a latency data point (grouped by tool name)
const createLatencyDataPoint = (toolName: string, avgLatency: number) => ({
  metric_name: SpanMetricKey.LATENCY,
  dimensions: {
    [SpanDimensionKey.SPAN_NAME]: toolName,
  },
  values: { [AggregationType.AVG]: avgLatency },
});

describe('ToolPerformanceSummary', () => {
  const testExperimentId = 'test-experiment-123';
  const startTimeMs = new Date('2025-12-22T10:00:00Z').getTime();
  const endTimeMs = new Date('2025-12-22T12:00:00Z').getTime();

  const defaultProps = {
    experimentId: testExperimentId,
    startTimeMs,
    endTimeMs,
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
          <ToolPerformanceSummary {...defaultProps} {...props} />
        </DesignSystemProvider>
      </QueryClientProvider>,
    );
  };

  // Handler returns different responses based on metric_name in request body
  const setupTraceMetricsHandler = (countDataPoints: any[], latencyDataPoints: any[]) => {
    server.use(
      rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
        const body = await req.json();
        if (body.metric_name === SpanMetricKey.SPAN_COUNT) {
          return res(ctx.json({ data_points: countDataPoints }));
        }
        if (body.metric_name === SpanMetricKey.LATENCY) {
          return res(ctx.json({ data_points: latencyDataPoints }));
        }
        return res(ctx.json({ data_points: [] }));
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
      expect(screen.queryByText('Tool Performance Summary')).not.toBeInTheDocument();
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
  });

  describe('with data', () => {
    const mockCountData = [
      createCountDataPoint('delivery_estimate', SpanStatus.OK, 1200),
      createCountDataPoint('delivery_estimate', SpanStatus.ERROR, 34),
      createCountDataPoint('get_order_status', SpanStatus.OK, 980),
      createCountDataPoint('get_order_status', SpanStatus.ERROR, 7),
      createCountDataPoint('calculate_shipping', SpanStatus.OK, 740),
      createCountDataPoint('calculate_shipping', SpanStatus.ERROR, 16),
    ];

    const mockLatencyData = [
      createLatencyDataPoint('delivery_estimate', 450),
      createLatencyDataPoint('get_order_status', 320),
      createLatencyDataPoint('calculate_shipping', 580),
    ];

    it('should display the section title', async () => {
      setupTraceMetricsHandler(mockCountData, mockLatencyData);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Tool Performance Summary')).toBeInTheDocument();
      });
    });

    it('should display table column headers', async () => {
      setupTraceMetricsHandler(mockCountData, mockLatencyData);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Tool')).toBeInTheDocument();
        expect(screen.getByText('Calls')).toBeInTheDocument();
        expect(screen.getByText('Success')).toBeInTheDocument();
        expect(screen.getByText('Latency')).toBeInTheDocument();
      });
    });

    it('should display tool names', async () => {
      setupTraceMetricsHandler(mockCountData, mockLatencyData);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('delivery_estimate')).toBeInTheDocument();
        expect(screen.getByText('get_order_status')).toBeInTheDocument();
        expect(screen.getByText('calculate_shipping')).toBeInTheDocument();
      });
    });

    it('should display formatted call counts', async () => {
      setupTraceMetricsHandler(mockCountData, mockLatencyData);

      renderComponent();

      await waitFor(() => {
        // 1234 total calls for delivery_estimate (1200 + 34)
        expect(screen.getByText('1.23K')).toBeInTheDocument();
      });
    });

    it('should display success rate percentages', async () => {
      setupTraceMetricsHandler(mockCountData, mockLatencyData);

      renderComponent();

      await waitFor(() => {
        // delivery_estimate: 1200/1234 = 97.24%
        expect(screen.getByText('97.24%')).toBeInTheDocument();
      });
    });

    it('should display formatted latencies', async () => {
      setupTraceMetricsHandler(mockCountData, mockLatencyData);

      renderComponent();

      await waitFor(() => {
        // 450ms latency
        expect(screen.getByText('450.00ms')).toBeInTheDocument();
        // 320ms latency
        expect(screen.getByText('320.00ms')).toBeInTheDocument();
      });
    });

    it('should sort tools by call count descending', async () => {
      setupTraceMetricsHandler(mockCountData, mockLatencyData);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('delivery_estimate')).toBeInTheDocument();
      });

      // Get all tool name elements and verify order
      const toolNames = screen.getAllByText(/delivery_estimate|get_order_status|calculate_shipping/);
      expect(toolNames[0].textContent).toBe('delivery_estimate'); // 1234 calls
      expect(toolNames[1].textContent).toBe('get_order_status'); // 987 calls
      expect(toolNames[2].textContent).toBe('calculate_shipping'); // 756 calls
    });
  });

  describe('with single tool', () => {
    it('should render table with single tool data', async () => {
      setupTraceMetricsHandler(
        [createCountDataPoint('single_tool', SpanStatus.OK, 500)],
        [createLatencyDataPoint('single_tool', 100)],
      );

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('single_tool')).toBeInTheDocument();
      });

      expect(screen.getByText('500')).toBeInTheDocument();
      expect(screen.getByText('100.00%')).toBeInTheDocument();
      expect(screen.getByText('100.00ms')).toBeInTheDocument();
    });
  });

  describe('latency formatting', () => {
    it('should format latencies over 1000ms as seconds', async () => {
      setupTraceMetricsHandler(
        [createCountDataPoint('slow_tool', SpanStatus.OK, 100)],
        [createLatencyDataPoint('slow_tool', 2500)], // 2.5 seconds
      );

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('2.50s')).toBeInTheDocument();
      });
    });
  });
});
