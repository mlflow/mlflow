import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useToolErrorRateChartData } from './useToolErrorRateChartData';
import {
  AggregationType,
  SpanMetricKey,
  SpanDimensionKey,
  SpanStatus,
} from '@databricks/web-shared/model-trace-explorer';
import type { ReactNode } from 'react';
import { setupServer } from '../../../../common/utils/setup-msw';
import { rest } from 'msw';
import { OverviewChartProvider } from '../OverviewChartContext';

// Helper to create a span count data point with status
const createSpanCountDataPoint = (timeBucket: string, status: string, count: number) => ({
  metric_name: SpanMetricKey.SPAN_COUNT,
  dimensions: {
    time_bucket: timeBucket,
    [SpanDimensionKey.SPAN_STATUS]: status,
  },
  values: { [AggregationType.COUNT]: count },
});

describe('useToolErrorRateChartData', () => {
  const testExperimentId = 'test-experiment-123';
  const startTimeMs = new Date('2025-12-22T10:00:00Z').getTime();
  const endTimeMs = new Date('2025-12-22T12:00:00Z').getTime();
  const timeIntervalSeconds = 3600; // 1 hour

  const timeBuckets = [
    new Date('2025-12-22T10:00:00Z').getTime(),
    new Date('2025-12-22T11:00:00Z').getTime(),
    new Date('2025-12-22T12:00:00Z').getTime(),
  ];

  const contextProps = {
    experimentIds: [testExperimentId],
    startTimeMs,
    endTimeMs,
    timeIntervalSeconds,
    timeBuckets,
  };

  const defaultToolName = 'test_tool';

  const server = setupServer();

  const createQueryClient = () =>
    new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

  const createWrapper = (contextOverrides: Partial<typeof contextProps> = {}) => {
    const queryClient = createQueryClient();
    const mergedContextProps = { ...contextProps, ...contextOverrides };
    return ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>
        <OverviewChartProvider {...mergedContextProps}>{children}</OverviewChartProvider>
      </QueryClientProvider>
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
    it('should return isLoading true while fetching', async () => {
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', (_req, res, ctx) => {
          return res(ctx.delay('infinite'));
        }),
      );

      const { result } = renderHook(() => useToolErrorRateChartData({ toolName: defaultToolName }), {
        wrapper: createWrapper(),
      });

      expect(result.current.isLoading).toBe(true);
      // chartData is computed from timeBuckets (from context), so it has entries even during loading
      expect(result.current.chartData).toHaveLength(3);
      expect(result.current.hasData).toBe(false);
    });
  });

  describe('error state', () => {
    it('should return error when API call fails', async () => {
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', (_req, res, ctx) => {
          return res(ctx.status(500), ctx.json({ error: 'API Error' }));
        }),
      );

      const { result } = renderHook(() => useToolErrorRateChartData({ toolName: defaultToolName }), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.error).toBeTruthy();
      });
    });
  });

  describe('empty data', () => {
    it('should return hasData false when no data points', async () => {
      // Default handler returns empty array

      const { result } = renderHook(() => useToolErrorRateChartData({ toolName: defaultToolName }), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.hasData).toBe(false);
      // Chart data still has time bucket entries with 0 error rates
      expect(result.current.chartData).toHaveLength(3);
      expect(result.current.chartData[0]).toHaveProperty('errorRate', 0);
    });

    it('should return hasData false when time range is not provided', async () => {
      // Default handler returns empty array

      const { result } = renderHook(() => useToolErrorRateChartData({ toolName: defaultToolName }), {
        wrapper: createWrapper({
          startTimeMs: undefined,
          endTimeMs: undefined,
          timeBuckets: [],
        }),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.hasData).toBe(false);
      expect(result.current.chartData).toHaveLength(0);
    });
  });

  describe('data transformation', () => {
    it('should calculate error rate correctly for a single time bucket', async () => {
      // 10% error rate: 10 errors out of 100 total
      setupTraceMetricsHandler([
        createSpanCountDataPoint('2025-12-22T10:00:00Z', SpanStatus.OK, 90),
        createSpanCountDataPoint('2025-12-22T10:00:00Z', SpanStatus.ERROR, 10),
      ]);

      const { result } = renderHook(() => useToolErrorRateChartData({ toolName: defaultToolName }), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.hasData).toBe(true);
      expect(result.current.chartData[0]).toHaveProperty('errorRate', 10);
    });

    it('should calculate error rate for multiple time buckets', async () => {
      // First bucket: 10% error rate (10/100)
      // Second bucket: 25% error rate (25/100)
      // Third bucket: no data (should be 0)
      setupTraceMetricsHandler([
        createSpanCountDataPoint('2025-12-22T10:00:00Z', SpanStatus.OK, 90),
        createSpanCountDataPoint('2025-12-22T10:00:00Z', SpanStatus.ERROR, 10),
        createSpanCountDataPoint('2025-12-22T11:00:00Z', SpanStatus.OK, 75),
        createSpanCountDataPoint('2025-12-22T11:00:00Z', SpanStatus.ERROR, 25),
      ]);

      const { result } = renderHook(() => useToolErrorRateChartData({ toolName: defaultToolName }), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.chartData).toHaveLength(3);
      expect(result.current.chartData[0]).toHaveProperty('errorRate', 10);
      expect(result.current.chartData[1]).toHaveProperty('errorRate', 25);
      expect(result.current.chartData[2]).toHaveProperty('errorRate', 0);
    });

    it('should handle 100% error rate', async () => {
      setupTraceMetricsHandler([createSpanCountDataPoint('2025-12-22T10:00:00Z', SpanStatus.ERROR, 50)]);

      const { result } = renderHook(() => useToolErrorRateChartData({ toolName: defaultToolName }), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.chartData[0]).toHaveProperty('errorRate', 100);
    });

    it('should handle 0% error rate (all OK)', async () => {
      setupTraceMetricsHandler([createSpanCountDataPoint('2025-12-22T10:00:00Z', SpanStatus.OK, 100)]);

      const { result } = renderHook(() => useToolErrorRateChartData({ toolName: defaultToolName }), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.chartData[0]).toHaveProperty('errorRate', 0);
    });

    it('should fill missing time buckets with 0 error rate', async () => {
      // Only provide data for the first bucket
      setupTraceMetricsHandler([
        createSpanCountDataPoint('2025-12-22T10:00:00Z', SpanStatus.OK, 80),
        createSpanCountDataPoint('2025-12-22T10:00:00Z', SpanStatus.ERROR, 20),
      ]);

      const { result } = renderHook(() => useToolErrorRateChartData({ toolName: defaultToolName }), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Should have all 3 time buckets
      expect(result.current.chartData).toHaveLength(3);
      // First bucket has 20% error rate
      expect(result.current.chartData[0]).toHaveProperty('errorRate', 20);
      // Missing buckets should be filled with 0
      expect(result.current.chartData[1]).toHaveProperty('errorRate', 0);
      expect(result.current.chartData[2]).toHaveProperty('errorRate', 0);
    });

    it('should include formatted timestamp in chart data', async () => {
      setupTraceMetricsHandler([createSpanCountDataPoint('2025-12-22T10:00:00Z', SpanStatus.OK, 100)]);

      const { result } = renderHook(() => useToolErrorRateChartData({ toolName: defaultToolName }), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Each data point should have a name (formatted timestamp)
      expect(result.current.chartData[0]).toHaveProperty('name');
      expect(typeof result.current.chartData[0].name).toBe('string');
    });

    it('should skip data points with missing time_bucket', async () => {
      setupTraceMetricsHandler([
        {
          metric_name: SpanMetricKey.SPAN_COUNT,
          dimensions: {
            [SpanDimensionKey.SPAN_STATUS]: SpanStatus.ERROR,
            // Missing time_bucket
          },
          values: { [AggregationType.COUNT]: 100 },
        },
        createSpanCountDataPoint('2025-12-22T10:00:00Z', SpanStatus.OK, 100),
      ]);

      const { result } = renderHook(() => useToolErrorRateChartData({ toolName: defaultToolName }), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Should have data, but missing time_bucket data point is ignored
      expect(result.current.hasData).toBe(true);
      // First bucket should have 0% error rate (only OK data counted)
      expect(result.current.chartData[0]).toHaveProperty('errorRate', 0);
    });

    it('should handle data points with missing count value', async () => {
      setupTraceMetricsHandler([
        {
          metric_name: SpanMetricKey.SPAN_COUNT,
          dimensions: {
            time_bucket: '2025-12-22T10:00:00Z',
            [SpanDimensionKey.SPAN_STATUS]: SpanStatus.ERROR,
          },
          values: {}, // Missing COUNT value
        },
        createSpanCountDataPoint('2025-12-22T10:00:00Z', SpanStatus.OK, 100),
      ]);

      const { result } = renderHook(() => useToolErrorRateChartData({ toolName: defaultToolName }), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Missing count defaults to 0, so error rate is 0/100 = 0%
      expect(result.current.chartData[0]).toHaveProperty('errorRate', 0);
    });

    it('should return hasData true when there are data points', async () => {
      setupTraceMetricsHandler([createSpanCountDataPoint('2025-12-22T10:00:00Z', SpanStatus.OK, 100)]);

      const { result } = renderHook(() => useToolErrorRateChartData({ toolName: defaultToolName }), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.hasData).toBe(true);
    });
  });

  describe('API request', () => {
    it('should include SPAN_STATUS dimension in request', async () => {
      let capturedBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useToolErrorRateChartData({ toolName: defaultToolName }), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(capturedBody).not.toBeNull();
      });

      expect(capturedBody.dimensions).toContain(SpanDimensionKey.SPAN_STATUS);
    });

    it('should filter for TOOL type spans', async () => {
      let capturedBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useToolErrorRateChartData({ toolName: defaultToolName }), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(capturedBody).not.toBeNull();
      });

      expect(capturedBody.filters).toContainEqual('span.type = "TOOL"');
    });

    it('should filter for specific tool name', async () => {
      let capturedBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useToolErrorRateChartData({ toolName: 'my_custom_tool' }), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(capturedBody).not.toBeNull();
      });

      expect(capturedBody.filters).toContainEqual('span.name = "my_custom_tool"');
    });

    it('should request COUNT aggregation for span count', async () => {
      let capturedBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useToolErrorRateChartData({ toolName: defaultToolName }), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(capturedBody).not.toBeNull();
      });

      expect(capturedBody.aggregations).toContainEqual({ aggregation_type: AggregationType.COUNT });
    });

    it('should include time interval for time bucketing', async () => {
      let capturedBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useToolErrorRateChartData({ toolName: defaultToolName }), {
        wrapper: createWrapper({ timeIntervalSeconds: 1800 }),
      });

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

      renderHook(() => useToolErrorRateChartData({ toolName: defaultToolName }), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(capturedBody).not.toBeNull();
      });

      expect(capturedBody.start_time_ms).toBe(startTimeMs);
      expect(capturedBody.end_time_ms).toBe(endTimeMs);
    });
  });
});
