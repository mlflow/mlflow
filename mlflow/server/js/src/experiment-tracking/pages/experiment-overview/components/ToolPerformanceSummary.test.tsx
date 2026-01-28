import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
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
import { OverviewChartProvider } from '../OverviewChartContext';

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
  const timeIntervalSeconds = 3600;
  const timeBuckets = [startTimeMs, startTimeMs + 3600000, endTimeMs];

  const contextProps = {
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

  const renderComponent = () => {
    const queryClient = createQueryClient();
    return renderWithIntl(
      <QueryClientProvider client={queryClient}>
        <DesignSystemProvider>
          <OverviewChartProvider {...contextProps}>
            <ToolPerformanceSummary />
          </OverviewChartProvider>
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
        expect(screen.getByText('Latency (AVG)')).toBeInTheDocument();
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

  describe('sorting functionality', () => {
    const mockCountData = [
      createCountDataPoint('alpha_tool', SpanStatus.OK, 500),
      createCountDataPoint('alpha_tool', SpanStatus.ERROR, 50),
      createCountDataPoint('beta_tool', SpanStatus.OK, 900),
      createCountDataPoint('beta_tool', SpanStatus.ERROR, 100),
      createCountDataPoint('gamma_tool', SpanStatus.OK, 200),
      createCountDataPoint('gamma_tool', SpanStatus.ERROR, 10),
    ];

    const mockLatencyData = [
      createLatencyDataPoint('alpha_tool', 300),
      createLatencyDataPoint('beta_tool', 100),
      createLatencyDataPoint('gamma_tool', 500),
    ];

    it('should sort by calls descending by default', async () => {
      setupTraceMetricsHandler(mockCountData, mockLatencyData);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('beta_tool')).toBeInTheDocument();
      });

      const toolNames = screen.getAllByText(/alpha_tool|beta_tool|gamma_tool/);
      expect(toolNames[0].textContent).toBe('beta_tool'); // 1000 calls
      expect(toolNames[1].textContent).toBe('alpha_tool'); // 550 calls
      expect(toolNames[2].textContent).toBe('gamma_tool'); // 210 calls
    });

    it('should toggle sort direction when clicking the same column', async () => {
      setupTraceMetricsHandler(mockCountData, mockLatencyData);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('beta_tool')).toBeInTheDocument();
      });

      // Click Calls header to toggle to ascending
      const callsHeader = screen.getByRole('button', { name: /Calls/i });
      await userEvent.click(callsHeader);

      // Now should be ascending - gamma_tool first (210 calls)
      const toolNamesAsc = screen.getAllByText(/alpha_tool|beta_tool|gamma_tool/);
      expect(toolNamesAsc[0].textContent).toBe('gamma_tool');
      expect(toolNamesAsc[1].textContent).toBe('alpha_tool');
      expect(toolNamesAsc[2].textContent).toBe('beta_tool');
    });

    it('should sort by tool name when clicking Tool header', async () => {
      setupTraceMetricsHandler(mockCountData, mockLatencyData);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('beta_tool')).toBeInTheDocument();
      });

      // Click Tool header
      const toolHeader = screen.getByRole('button', { name: /^Tool$/i });
      await userEvent.click(toolHeader);

      // Should sort by name descending first (gamma > beta > alpha)
      const toolNames = screen.getAllByText(/alpha_tool|beta_tool|gamma_tool/);
      expect(toolNames[0].textContent).toBe('gamma_tool');
      expect(toolNames[1].textContent).toBe('beta_tool');
      expect(toolNames[2].textContent).toBe('alpha_tool');
    });

    it('should sort by success rate when clicking Success header', async () => {
      setupTraceMetricsHandler(mockCountData, mockLatencyData);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('beta_tool')).toBeInTheDocument();
      });

      // Click Success header
      const successHeader = screen.getByRole('button', { name: /Success/i });
      await userEvent.click(successHeader);

      // Should sort by success rate descending
      // gamma_tool: 200/210 = 95.24%
      // alpha_tool: 500/550 = 90.91%
      // beta_tool: 900/1000 = 90%
      const toolNames = screen.getAllByText(/alpha_tool|beta_tool|gamma_tool/);
      expect(toolNames[0].textContent).toBe('gamma_tool');
      expect(toolNames[1].textContent).toBe('alpha_tool');
      expect(toolNames[2].textContent).toBe('beta_tool');
    });

    it('should sort by latency when clicking Latency header', async () => {
      setupTraceMetricsHandler(mockCountData, mockLatencyData);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('beta_tool')).toBeInTheDocument();
      });

      // Click Latency header
      const latencyHeader = screen.getByRole('button', { name: /Latency \(AVG\)/i });
      await userEvent.click(latencyHeader);

      // Should sort by latency descending
      // gamma_tool: 500ms, alpha_tool: 300ms, beta_tool: 100ms
      const toolNames = screen.getAllByText(/alpha_tool|beta_tool|gamma_tool/);
      expect(toolNames[0].textContent).toBe('gamma_tool');
      expect(toolNames[1].textContent).toBe('alpha_tool');
      expect(toolNames[2].textContent).toBe('beta_tool');
    });

    it('should support keyboard navigation for sorting', async () => {
      setupTraceMetricsHandler(mockCountData, mockLatencyData);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('beta_tool')).toBeInTheDocument();
      });

      // Focus and press Enter on Tool header
      const toolHeader = screen.getByRole('button', { name: /^Tool$/i });
      toolHeader.focus();
      await userEvent.keyboard('{Enter}');

      // Should sort by name descending
      const toolNames = screen.getAllByText(/alpha_tool|beta_tool|gamma_tool/);
      expect(toolNames[0].textContent).toBe('gamma_tool');
      expect(toolNames[1].textContent).toBe('beta_tool');
      expect(toolNames[2].textContent).toBe('alpha_tool');
    });

    it('should display sort icon on active column', async () => {
      setupTraceMetricsHandler(mockCountData, mockLatencyData);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('beta_tool')).toBeInTheDocument();
      });

      // Default is Calls descending - check for descending icon
      const callsHeader = screen.getByRole('button', { name: /Calls/i });
      expect(within(callsHeader).getByRole('img', { hidden: true })).toBeInTheDocument();

      // Tool header should not have sort icon
      const toolHeader = screen.getByRole('button', { name: /^Tool$/i });
      expect(within(toolHeader).queryByRole('img', { hidden: true })).not.toBeInTheDocument();
    });
  });

  describe('scroll to chart functionality', () => {
    it('should make tool names clickable', async () => {
      setupTraceMetricsHandler(
        [createCountDataPoint('test_tool', SpanStatus.OK, 100)],
        [createLatencyDataPoint('test_tool', 200)],
      );

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('test_tool')).toBeInTheDocument();
      });

      // Tool name should be in a clickable element
      const toolCell = screen.getByText('test_tool').closest('[role="button"]');
      expect(toolCell).toBeInTheDocument();
    });

    it('should scroll to chart when tool name is clicked', async () => {
      // Create a mock element to scroll to
      const mockElement = document.createElement('div');
      mockElement.id = 'tool-chart-click_tool';
      mockElement.scrollIntoView = jest.fn();
      document.body.appendChild(mockElement);

      setupTraceMetricsHandler(
        [createCountDataPoint('click_tool', SpanStatus.OK, 100)],
        [createLatencyDataPoint('click_tool', 200)],
      );

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('click_tool')).toBeInTheDocument();
      });

      const toolCell = screen.getByText('click_tool').closest('[role="button"]');
      await userEvent.click(toolCell!);

      expect(mockElement.scrollIntoView).toHaveBeenCalledWith({ behavior: 'smooth', block: 'start' });

      // Cleanup
      document.body.removeChild(mockElement);
    });
  });
});
