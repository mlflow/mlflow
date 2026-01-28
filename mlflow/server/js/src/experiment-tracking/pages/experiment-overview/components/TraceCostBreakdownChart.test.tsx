import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { TraceCostBreakdownChart } from './TraceCostBreakdownChart';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import {
  MetricViewType,
  AggregationType,
  SpanMetricKey,
  SpanDimensionKey,
} from '@databricks/web-shared/model-trace-explorer';
import { setupServer } from '../../../../common/utils/setup-msw';
import { rest } from 'msw';
import { OverviewChartProvider } from '../OverviewChartContext';

// Helper to create a cost breakdown data point
const createCostDataPoint = (modelName: string, totalCost: number) => ({
  metric_name: SpanMetricKey.TOTAL_COST,
  dimensions: { [SpanDimensionKey.MODEL_NAME]: modelName },
  values: { [AggregationType.SUM]: totalCost },
});

describe('TraceCostBreakdownChart', () => {
  const testExperimentId = 'test-experiment-123';
  const startTimeMs = new Date('2025-12-22T10:00:00Z').getTime();
  const endTimeMs = new Date('2025-12-22T12:00:00Z').getTime();
  const timeIntervalSeconds = 3600;

  const timeBuckets = [
    new Date('2025-12-22T10:00:00Z').getTime(),
    new Date('2025-12-22T11:00:00Z').getTime(),
    new Date('2025-12-22T12:00:00Z').getTime(),
  ];

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
            <TraceCostBreakdownChart />
          </OverviewChartProvider>
        </DesignSystemProvider>
      </QueryClientProvider>,
    );
  };

  const setupTraceMetricsHandler = (dataPoints: any[]) => {
    server.use(
      rest.post('ajax-api/3.0/mlflow/traces/metrics', async (_req, res, ctx) => {
        return res(ctx.json({ data_points: dataPoints }));
      }),
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
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
      expect(screen.queryByText('Cost Breakdown')).not.toBeInTheDocument();
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
        expect(screen.getByText('No cost data available')).toBeInTheDocument();
      });
    });

    it('should render empty state when all costs are zero', async () => {
      setupTraceMetricsHandler([createCostDataPoint('gpt-4', 0), createCostDataPoint('gpt-3.5', 0)]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('No cost data available')).toBeInTheDocument();
      });
    });
  });

  describe('with data', () => {
    const mockDataPoints = [
      createCostDataPoint('gpt-4', 0.05),
      createCostDataPoint('gpt-3.5-turbo', 0.03),
      createCostDataPoint('claude-3', 0.02),
    ];

    it('should render the chart title', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Cost Breakdown')).toBeInTheDocument();
      });
    });

    it('should render the subtitle', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Total Cost')).toBeInTheDocument();
      });
    });

    it('should display total cost formatted as USD', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      // Total = 0.05 + 0.03 + 0.02 = 0.10
      await waitFor(() => {
        expect(screen.getByText('$0.10')).toBeInTheDocument();
      });
    });

    it('should render pie chart component', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('pie-chart')).toBeInTheDocument();
      });
    });

    it('should render pie with correct number of data points', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('pie')).toHaveAttribute('data-count', '3');
      });
    });

    it('should render legend component', async () => {
      setupTraceMetricsHandler(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('legend')).toBeInTheDocument();
      });
    });

    it('should format large costs correctly', async () => {
      setupTraceMetricsHandler([createCostDataPoint('gpt-4', 1234.56)]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('$1,234.56')).toBeInTheDocument();
      });
    });

    it('should format small costs with proper precision', async () => {
      setupTraceMetricsHandler([createCostDataPoint('gpt-4', 0.000123)]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('$0.000123')).toBeInTheDocument();
      });
    });
  });

  describe('API call parameters', () => {
    it('should call API with correct parameters for cost breakdown', async () => {
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
          view_type: MetricViewType.SPANS,
          metric_name: SpanMetricKey.TOTAL_COST,
          aggregations: [{ aggregation_type: AggregationType.SUM }],
          dimensions: [SpanDimensionKey.MODEL_NAME],
        });
      });
    });

    it('should include time range in API request', async () => {
      let capturedRequest: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedRequest = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderComponent();

      await waitFor(() => {
        expect(capturedRequest?.start_time_ms).toBe(startTimeMs);
        expect(capturedRequest?.end_time_ms).toBe(endTimeMs);
      });
    });
  });
});
