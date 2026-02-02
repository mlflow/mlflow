import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { ToolCallStatistics } from './ToolCallStatistics';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import {
  MetricViewType,
  AggregationType,
  SpanMetricKey,
  SpanFilterKey,
  SpanType,
  SpanDimensionKey,
} from '@databricks/web-shared/model-trace-explorer';
import { setupServer } from '../../../../common/utils/setup-msw';
import { rest } from 'msw';
import { OverviewChartProvider } from '../OverviewChartContext';

// Helper to create a count data point grouped by status
const createCountByStatusDataPoint = (status: string, count: number) => ({
  metric_name: SpanMetricKey.SPAN_COUNT,
  dimensions: { [SpanDimensionKey.SPAN_STATUS]: status },
  values: { [AggregationType.COUNT]: count },
});

// Helper to create a latency data point
const createLatencyDataPoint = (avgLatency: number) => ({
  metric_name: SpanMetricKey.LATENCY,
  dimensions: {},
  values: { [AggregationType.AVG]: avgLatency },
});

describe('ToolCallStatistics', () => {
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
            <ToolCallStatistics />
          </OverviewChartProvider>
        </DesignSystemProvider>
      </QueryClientProvider>,
    );
  };

  // Helper to setup MSW handler that returns different responses based on metric_name
  const setupTraceMetricsHandler = (countDataPoints: any[], latencyDataPoints: any[]) => {
    server.use(
      rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
        const body = await req.json();
        if (body.metric_name === SpanMetricKey.LATENCY) {
          return res(ctx.json({ data_points: latencyDataPoints }));
        }
        return res(ctx.json({ data_points: countDataPoints }));
      }),
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
    // Default: return empty data points
    setupTraceMetricsHandler([], []);
  });

  describe('loading state', () => {
    it('should render loading skeletons while data is being fetched', () => {
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', (_req, res, ctx) => {
          return res(ctx.delay('infinite'));
        }),
      );

      renderComponent();

      // Should show stat card labels but values should be replaced with skeletons
      expect(screen.getByText('Total Tool Calls')).toBeInTheDocument();
      expect(screen.getByText('Success Rate')).toBeInTheDocument();
      expect(screen.getByText('Avg Latency')).toBeInTheDocument();
      expect(screen.getByText('Failed Calls')).toBeInTheDocument();

      // Values should not be displayed during loading (replaced with skeletons)
      expect(screen.queryByText('0')).not.toBeInTheDocument();
      expect(screen.queryByText('0.00%')).not.toBeInTheDocument();
      expect(screen.queryByText('0.00ms')).not.toBeInTheDocument();
    });
  });

  describe('with data', () => {
    it('should render all four stat cards', async () => {
      const countData = [createCountByStatusDataPoint('OK', 100), createCountByStatusDataPoint('ERROR', 5)];
      const latencyData = [createLatencyDataPoint(250)];
      setupTraceMetricsHandler(countData, latencyData);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Total Tool Calls')).toBeInTheDocument();
        expect(screen.getByText('Success Rate')).toBeInTheDocument();
        expect(screen.getByText('Avg Latency')).toBeInTheDocument();
        expect(screen.getByText('Failed Calls')).toBeInTheDocument();
      });
    });

    it('should display correct total count', async () => {
      const countData = [createCountByStatusDataPoint('OK', 100), createCountByStatusDataPoint('ERROR', 5)];
      setupTraceMetricsHandler(countData, [createLatencyDataPoint(0)]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('105')).toBeInTheDocument(); // 100 + 5 = 105
      });
    });

    it('should display correct success rate', async () => {
      const countData = [createCountByStatusDataPoint('OK', 95), createCountByStatusDataPoint('ERROR', 5)];
      setupTraceMetricsHandler(countData, [createLatencyDataPoint(0)]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('95.00%')).toBeInTheDocument(); // 95/100 = 95%
      });
    });

    it('should display correct failed count', async () => {
      const countData = [createCountByStatusDataPoint('OK', 90), createCountByStatusDataPoint('ERROR', 10)];
      setupTraceMetricsHandler(countData, [createLatencyDataPoint(0)]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('10')).toBeInTheDocument(); // 10 errors
      });
    });

    it('should display latency in milliseconds when < 1000ms', async () => {
      setupTraceMetricsHandler([createCountByStatusDataPoint('OK', 100)], [createLatencyDataPoint(250.5)]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('250.50ms')).toBeInTheDocument();
      });
    });

    it('should display latency in seconds when >= 1000ms', async () => {
      setupTraceMetricsHandler([createCountByStatusDataPoint('OK', 100)], [createLatencyDataPoint(1500)]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('1.50s')).toBeInTheDocument();
      });
    });

    it('should format large numbers with K suffix', async () => {
      const countData = [createCountByStatusDataPoint('OK', 5000)];
      setupTraceMetricsHandler(countData, [createLatencyDataPoint(0)]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('5.00K')).toBeInTheDocument();
      });
    });

    it('should handle UNSET status in addition to OK and ERROR', async () => {
      const countData = [
        createCountByStatusDataPoint('OK', 80),
        createCountByStatusDataPoint('ERROR', 15),
        createCountByStatusDataPoint('UNSET', 5),
      ];
      setupTraceMetricsHandler(countData, [createLatencyDataPoint(0)]);

      renderComponent();

      await waitFor(() => {
        // Total should include all statuses (80 + 15 + 5 = 100)
        expect(screen.getByText('100')).toBeInTheDocument();
        // Success rate based on OK only (80/100 = 80%)
        expect(screen.getByText('80.00%')).toBeInTheDocument();
        // Failed count based on ERROR only (15)
        expect(screen.getByText('15')).toBeInTheDocument();
      });
    });
  });

  describe('empty data', () => {
    it('should display zeros when no data is returned', async () => {
      // Default handler already returns empty arrays
      renderComponent();

      await waitFor(() => {
        // Should show 0 for counts
        const zeros = screen.getAllByText('0');
        expect(zeros.length).toBeGreaterThanOrEqual(2); // Total and Failed
        // Should show 0.00% for success rate
        expect(screen.getByText('0.00%')).toBeInTheDocument();
        // Should show 0.00ms for latency
        expect(screen.getByText('0.00ms')).toBeInTheDocument();
      });
    });
  });

  describe('error state', () => {
    it('should render error state when API call fails', async () => {
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

    it('should render error state when counts API call fails', async () => {
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          const body = await req.json();
          if (body.metric_name === SpanMetricKey.LATENCY) {
            return res(ctx.json({ data_points: [createLatencyDataPoint(100)] }));
          }
          return res(ctx.status(500), ctx.json({ error: 'Counts API Error' }));
        }),
      );

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Failed to load chart data')).toBeInTheDocument();
      });
    });

    it('should render error state when latency API call fails', async () => {
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          const body = await req.json();
          if (body.metric_name === SpanMetricKey.LATENCY) {
            return res(ctx.status(500), ctx.json({ error: 'Latency API Error' }));
          }
          return res(ctx.json({ data_points: [createCountByStatusDataPoint('OK', 100)] }));
        }),
      );

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Failed to load chart data')).toBeInTheDocument();
      });
    });
  });

  describe('API call parameters', () => {
    it('should call API with correct parameters for counts query', async () => {
      let capturedCountsBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          const body = await req.json();
          if (body.metric_name === SpanMetricKey.SPAN_COUNT) {
            capturedCountsBody = body;
          }
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent();

      await waitFor(() => {
        expect(capturedCountsBody).not.toBeNull();
      });

      expect(capturedCountsBody).toMatchObject({
        experiment_ids: [testExperimentId],
        view_type: MetricViewType.SPANS,
        metric_name: SpanMetricKey.SPAN_COUNT,
        aggregations: [{ aggregation_type: AggregationType.COUNT }],
        filters: [`span.${SpanFilterKey.TYPE} = "${SpanType.TOOL}"`],
        dimensions: [SpanDimensionKey.SPAN_STATUS],
      });
    });

    it('should call API with correct parameters for latency query', async () => {
      let capturedLatencyBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          const body = await req.json();
          if (body.metric_name === SpanMetricKey.LATENCY) {
            capturedLatencyBody = body;
          }
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent();

      await waitFor(() => {
        expect(capturedLatencyBody).not.toBeNull();
      });

      expect(capturedLatencyBody).toMatchObject({
        experiment_ids: [testExperimentId],
        view_type: MetricViewType.SPANS,
        metric_name: SpanMetricKey.LATENCY,
        aggregations: [{ aggregation_type: AggregationType.AVG }],
        filters: [`span.${SpanFilterKey.TYPE} = "${SpanType.TOOL}"`],
      });
    });

    it('should include time range in API calls', async () => {
      let capturedBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          const body = await req.json();
          if (!capturedBody) {
            capturedBody = body;
          }
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent();

      await waitFor(() => {
        expect(capturedBody).not.toBeNull();
      });

      expect(capturedBody.start_time_ms).toBe(startTimeMs);
      expect(capturedBody.end_time_ms).toBe(endTimeMs);
    });
  });
});
