import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { ToolErrorRateChart } from './ToolErrorRateChart';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import {
  MetricViewType,
  AggregationType,
  SpanMetricKey,
  SpanFilterKey,
  SpanType,
  SpanStatus,
  SpanDimensionKey,
  TIME_BUCKET_DIMENSION_KEY,
} from '@databricks/web-shared/model-trace-explorer';
import { setupServer } from '../../../../common/utils/setup-msw';
import { rest } from 'msw';

// Helper to create a data point with time bucket and status
const createDataPoint = (timeBucket: string, status: string, count: number) => ({
  metric_name: SpanMetricKey.SPAN_COUNT,
  dimensions: {
    [TIME_BUCKET_DIMENSION_KEY]: timeBucket,
    [SpanDimensionKey.SPAN_STATUS]: status,
  },
  values: { [AggregationType.COUNT]: count },
});

describe('ToolErrorRateChart', () => {
  const testExperimentId = 'test-experiment-123';
  const startTimeMs = new Date('2025-12-22T10:00:00Z').getTime();
  const endTimeMs = new Date('2025-12-22T12:00:00Z').getTime();
  const timeIntervalSeconds = 3600; // 1 hour
  const timeBuckets = [new Date('2025-12-22T10:00:00Z').getTime(), new Date('2025-12-22T11:00:00Z').getTime()];

  const defaultProps = {
    experimentId: testExperimentId,
    startTimeMs,
    endTimeMs,
    timeIntervalSeconds,
    timeBuckets,
    toolName: 'get_weather',
    overallErrorRate: 10.5,
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
          <ToolErrorRateChart {...defaultProps} {...props} />
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
    it('should render loading skeleton while data is being fetched', () => {
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', (_req, res, ctx) => {
          return res(ctx.delay('infinite'));
        }),
      );

      renderComponent();

      // Check that actual chart content is not rendered during loading
      expect(screen.queryByText('get_weather')).not.toBeInTheDocument();
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
  });

  describe('empty state', () => {
    it('should render empty state when no data is returned', async () => {
      // Default handler returns empty array
      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('No data available for the selected time range')).toBeInTheDocument();
      });
    });
  });

  describe('with data', () => {
    it('should render the tool name as title', async () => {
      setupTraceMetricsHandler([createDataPoint('2025-12-22T10:00:00Z', SpanStatus.OK, 100)]);

      renderComponent({ toolName: 'search_documentation' });

      await waitFor(() => {
        expect(screen.getByText('search_documentation')).toBeInTheDocument();
      });
    });

    it('should display overall error rate in header', async () => {
      setupTraceMetricsHandler([createDataPoint('2025-12-22T10:00:00Z', SpanStatus.OK, 100)]);

      renderComponent({ overallErrorRate: 25.5 });

      await waitFor(() => {
        expect(screen.getByText('25.50%')).toBeInTheDocument();
        expect(screen.getByText('overall error rate')).toBeInTheDocument();
      });
    });

    it('should render the line chart', async () => {
      setupTraceMetricsHandler([
        createDataPoint('2025-12-22T10:00:00Z', SpanStatus.OK, 90),
        createDataPoint('2025-12-22T10:00:00Z', SpanStatus.ERROR, 10),
      ]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
        expect(screen.getByTestId('line-Error Rate')).toBeInTheDocument();
      });
    });

    it('should display "Over time" label', async () => {
      setupTraceMetricsHandler([createDataPoint('2025-12-22T10:00:00Z', SpanStatus.OK, 100)]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Over time')).toBeInTheDocument();
      });
    });
  });

  describe('API call parameters', () => {
    it('should call API with correct view type and metric name', async () => {
      let capturedBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent();

      await waitFor(() => {
        expect(capturedBody).not.toBeNull();
      });

      expect(capturedBody).toMatchObject({
        experiment_ids: [testExperimentId],
        view_type: MetricViewType.SPANS,
        metric_name: SpanMetricKey.SPAN_COUNT,
        aggregations: [{ aggregation_type: AggregationType.COUNT }],
      });
    });

    it('should filter by tool type and tool name', async () => {
      let capturedBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent({ toolName: 'my_custom_tool' });

      await waitFor(() => {
        expect(capturedBody).not.toBeNull();
      });

      expect(capturedBody.filters).toContain(`span.${SpanFilterKey.TYPE} = "${SpanType.TOOL}"`);
      expect(capturedBody.filters).toContain(`span.${SpanFilterKey.NAME} = "my_custom_tool"`);
    });

    it('should include span_status dimension', async () => {
      let capturedBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent();

      await waitFor(() => {
        expect(capturedBody).not.toBeNull();
      });

      expect(capturedBody.dimensions).toContain(SpanDimensionKey.SPAN_STATUS);
    });

    it('should include time interval for time bucketing', async () => {
      let capturedBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent({ timeIntervalSeconds: 1800 }); // 30 minutes

      await waitFor(() => {
        expect(capturedBody).not.toBeNull();
      });

      expect(capturedBody.time_interval_seconds).toBe(1800);
    });

    it('should include time range in API call', async () => {
      let capturedBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
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

  describe('data transformation', () => {
    it('should calculate error rate correctly per time bucket', async () => {
      // 10% error rate in first bucket (10 errors out of 100 total)
      // 20% error rate in second bucket (20 errors out of 100 total)
      setupTraceMetricsHandler([
        createDataPoint('2025-12-22T10:00:00Z', SpanStatus.OK, 90),
        createDataPoint('2025-12-22T10:00:00Z', SpanStatus.ERROR, 10),
        createDataPoint('2025-12-22T11:00:00Z', SpanStatus.OK, 80),
        createDataPoint('2025-12-22T11:00:00Z', SpanStatus.ERROR, 20),
      ]);

      renderComponent();

      await waitFor(() => {
        // The chart should render with data
        const chart = screen.getByTestId('line-chart');
        expect(chart).toHaveAttribute('data-count', '2'); // 2 time buckets
      });
    });

    it('should handle missing time buckets by filling with 0', async () => {
      // Only one bucket has data
      setupTraceMetricsHandler([createDataPoint('2025-12-22T10:00:00Z', SpanStatus.OK, 100)]);

      renderComponent();

      await waitFor(() => {
        // Chart should have data for all time buckets (filled with 0 for missing)
        const chart = screen.getByTestId('line-chart');
        expect(chart).toHaveAttribute('data-count', '2'); // Both buckets
      });
    });
  });
});
