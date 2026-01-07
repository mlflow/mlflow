import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { ToolCallChartsSection } from './ToolCallChartsSection';
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
} from '@databricks/web-shared/model-trace-explorer';
import { setupServer } from '../../../../common/utils/setup-msw';
import { rest } from 'msw';

// Helper to create a data point with tool name and status
const createDataPoint = (toolName: string, status: string, count: number) => ({
  metric_name: SpanMetricKey.SPAN_COUNT,
  dimensions: {
    [SpanDimensionKey.SPAN_NAME]: toolName,
    [SpanDimensionKey.SPAN_STATUS]: status,
  },
  values: { [AggregationType.COUNT]: count },
});

describe('ToolCallChartsSection', () => {
  const testExperimentId = 'test-experiment-123';
  const startTimeMs = new Date('2025-12-22T10:00:00Z').getTime();
  const endTimeMs = new Date('2025-12-22T12:00:00Z').getTime();
  const timeIntervalSeconds = 3600;
  const timeBuckets = [new Date('2025-12-22T10:00:00Z').getTime(), new Date('2025-12-22T11:00:00Z').getTime()];

  const defaultProps: {
    experimentId: string;
    startTimeMs: number;
    endTimeMs: number;
    timeIntervalSeconds: number;
    timeBuckets: number[];
    searchQuery?: string;
  } = {
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

  const renderComponent = (props: Partial<typeof defaultProps> = {}) => {
    const queryClient = createQueryClient();
    return renderWithIntl(
      <QueryClientProvider client={queryClient}>
        <DesignSystemProvider>
          <ToolCallChartsSection {...defaultProps} {...props} />
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
      expect(screen.queryAllByTestId(/^tool-chart-/)).toHaveLength(0);
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
    it('should render nothing when no tools are found', async () => {
      // Default handler returns empty array
      const { container } = renderComponent();

      await waitFor(() => {
        // Component returns null when no data
        expect(container).toBeEmptyDOMElement();
      });
    });
  });

  describe('with data', () => {
    it('should render a chart for each tool', async () => {
      setupTraceMetricsHandler([
        createDataPoint('get_weather', SpanStatus.OK, 100),
        createDataPoint('search_docs', SpanStatus.OK, 50),
      ]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('tool-chart-get_weather')).toBeInTheDocument();
        expect(screen.getByTestId('tool-chart-search_docs')).toBeInTheDocument();
      });
    });

    it('should display tool names alphabetically sorted', async () => {
      setupTraceMetricsHandler([
        createDataPoint('zebra_tool', SpanStatus.OK, 10),
        createDataPoint('alpha_tool', SpanStatus.OK, 20),
        createDataPoint('middle_tool', SpanStatus.OK, 30),
      ]);

      renderComponent();

      await waitFor(() => {
        const charts = screen.getAllByTestId(/^tool-chart-/);
        expect(charts[0]).toHaveAttribute('data-testid', 'tool-chart-alpha_tool');
        expect(charts[1]).toHaveAttribute('data-testid', 'tool-chart-middle_tool');
        expect(charts[2]).toHaveAttribute('data-testid', 'tool-chart-zebra_tool');
      });
    });

    it('should calculate correct error rate for each tool', async () => {
      // get_weather: 10 errors / 100 total = 10%
      // search_docs: 25 errors / 100 total = 25%
      setupTraceMetricsHandler([
        createDataPoint('get_weather', SpanStatus.OK, 90),
        createDataPoint('get_weather', SpanStatus.ERROR, 10),
        createDataPoint('search_docs', SpanStatus.OK, 75),
        createDataPoint('search_docs', SpanStatus.ERROR, 25),
      ]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('10.00%')).toBeInTheDocument();
        expect(screen.getByText('25.00%')).toBeInTheDocument();
      });
    });

    it('should handle tools with 0% error rate', async () => {
      setupTraceMetricsHandler([createDataPoint('perfect_tool', SpanStatus.OK, 100)]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('0.00%')).toBeInTheDocument();
      });
    });

    it('should handle tools with 100% error rate', async () => {
      setupTraceMetricsHandler([createDataPoint('broken_tool', SpanStatus.ERROR, 50)]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('100.00%')).toBeInTheDocument();
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

    it('should filter by tool type', async () => {
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

      expect(capturedBody.filters).toContain(`span.${SpanFilterKey.TYPE} = "${SpanType.TOOL}"`);
    });

    it('should include span_name and span_status dimensions', async () => {
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

      expect(capturedBody.dimensions).toContain(SpanDimensionKey.SPAN_NAME);
      expect(capturedBody.dimensions).toContain(SpanDimensionKey.SPAN_STATUS);
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

  describe('searchQuery filtering', () => {
    it('should filter tools by searchQuery', async () => {
      setupTraceMetricsHandler([
        createDataPoint('get_weather', SpanStatus.OK, 100),
        createDataPoint('search_docs', SpanStatus.OK, 50),
        createDataPoint('fetch_data', SpanStatus.OK, 75),
      ]);

      renderComponent({ searchQuery: 'weather' });

      await waitFor(() => {
        expect(screen.getByTestId('tool-chart-get_weather')).toBeInTheDocument();
      });

      expect(screen.queryByTestId('tool-chart-search_docs')).not.toBeInTheDocument();
      expect(screen.queryByTestId('tool-chart-fetch_data')).not.toBeInTheDocument();
    });

    it('should filter tools case-insensitively', async () => {
      setupTraceMetricsHandler([
        createDataPoint('GetWeather', SpanStatus.OK, 100),
        createDataPoint('search_docs', SpanStatus.OK, 50),
      ]);

      renderComponent({ searchQuery: 'GETWEATHER' });

      await waitFor(() => {
        expect(screen.getByTestId('tool-chart-GetWeather')).toBeInTheDocument();
      });

      expect(screen.queryByTestId('tool-chart-search_docs')).not.toBeInTheDocument();
    });

    it('should show all tools when searchQuery is empty', async () => {
      setupTraceMetricsHandler([
        createDataPoint('get_weather', SpanStatus.OK, 100),
        createDataPoint('search_docs', SpanStatus.OK, 50),
      ]);

      renderComponent({ searchQuery: '' });

      await waitFor(() => {
        expect(screen.getByTestId('tool-chart-get_weather')).toBeInTheDocument();
        expect(screen.getByTestId('tool-chart-search_docs')).toBeInTheDocument();
      });
    });

    it('should show all tools when searchQuery is whitespace only', async () => {
      setupTraceMetricsHandler([
        createDataPoint('get_weather', SpanStatus.OK, 100),
        createDataPoint('search_docs', SpanStatus.OK, 50),
      ]);

      renderComponent({ searchQuery: '   ' });

      await waitFor(() => {
        expect(screen.getByTestId('tool-chart-get_weather')).toBeInTheDocument();
        expect(screen.getByTestId('tool-chart-search_docs')).toBeInTheDocument();
      });
    });

    it('should render nothing when searchQuery matches no tools', async () => {
      setupTraceMetricsHandler([
        createDataPoint('get_weather', SpanStatus.OK, 100),
        createDataPoint('search_docs', SpanStatus.OK, 50),
      ]);

      const { container } = renderComponent({ searchQuery: 'nonexistent' });

      await waitFor(() => {
        expect(container).toBeEmptyDOMElement();
      });
    });

    it('should support partial matching in searchQuery', async () => {
      setupTraceMetricsHandler([
        createDataPoint('get_weather', SpanStatus.OK, 100),
        createDataPoint('get_location', SpanStatus.OK, 50),
        createDataPoint('search_docs', SpanStatus.OK, 75),
      ]);

      renderComponent({ searchQuery: 'get_' });

      await waitFor(() => {
        expect(screen.getByTestId('tool-chart-get_weather')).toBeInTheDocument();
        expect(screen.getByTestId('tool-chart-get_location')).toBeInTheDocument();
      });

      expect(screen.queryByTestId('tool-chart-search_docs')).not.toBeInTheDocument();
    });
  });
});
