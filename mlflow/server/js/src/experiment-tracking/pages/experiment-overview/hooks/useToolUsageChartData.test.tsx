import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useToolUsageChartData } from './useToolUsageChartData';
import { AggregationType, SpanMetricKey, SpanDimensionKey } from '@databricks/web-shared/model-trace-explorer';
import type { ReactNode } from 'react';
import { setupServer } from '../../../../common/utils/setup-msw';
import { rest } from 'msw';

// Helper to create a tool usage data point
const createToolUsageDataPoint = (timeBucket: string, toolName: string, count: number) => ({
  metric_name: SpanMetricKey.SPAN_COUNT,
  dimensions: {
    time_bucket: timeBucket,
    [SpanDimensionKey.SPAN_NAME]: toolName,
  },
  values: { [AggregationType.COUNT]: count },
});

describe('useToolUsageChartData', () => {
  const testExperimentId = 'test-experiment-123';
  const startTimeMs = new Date('2025-12-22T10:00:00Z').getTime();
  const endTimeMs = new Date('2025-12-22T12:00:00Z').getTime();
  const timeIntervalSeconds = 3600; // 1 hour

  const timeBuckets = [
    new Date('2025-12-22T10:00:00Z').getTime(),
    new Date('2025-12-22T11:00:00Z').getTime(),
    new Date('2025-12-22T12:00:00Z').getTime(),
  ];

  const defaultProps = {
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

  const createWrapper = () => {
    const queryClient = createQueryClient();
    return ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
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

      const { result } = renderHook(() => useToolUsageChartData(defaultProps), {
        wrapper: createWrapper(),
      });

      expect(result.current.isLoading).toBe(true);
      expect(result.current.toolNames).toEqual([]);
    });
  });

  describe('error state', () => {
    it('should return error when API call fails', async () => {
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', (_req, res, ctx) => {
          return res(ctx.status(500), ctx.json({ error: 'API Error' }));
        }),
      );

      const { result } = renderHook(() => useToolUsageChartData(defaultProps), {
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

      const { result } = renderHook(() => useToolUsageChartData(defaultProps), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.hasData).toBe(false);
      // Chart data still has time bucket entries, just no tool values
      expect(result.current.chartData).toHaveLength(3);
      expect(result.current.toolNames).toEqual([]);
    });

    it('should return hasData false when time range is not provided', async () => {
      // Default handler returns empty array

      const { result } = renderHook(
        () =>
          useToolUsageChartData({
            ...defaultProps,
            startTimeMs: undefined,
            endTimeMs: undefined,
            timeBuckets: [],
          }),
        { wrapper: createWrapper() },
      );

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.hasData).toBe(false);
    });
  });

  describe('data transformation', () => {
    it('should extract unique tool names sorted alphabetically', async () => {
      setupTraceMetricsHandler([
        createToolUsageDataPoint('2025-12-22T10:00:00Z', 'zebra_tool', 10),
        createToolUsageDataPoint('2025-12-22T10:00:00Z', 'alpha_tool', 20),
        createToolUsageDataPoint('2025-12-22T11:00:00Z', 'beta_tool', 30),
      ]);

      const { result } = renderHook(() => useToolUsageChartData(defaultProps), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.toolNames).toEqual(['alpha_tool', 'beta_tool', 'zebra_tool']);
    });

    it('should create chart data with all time buckets', async () => {
      setupTraceMetricsHandler([
        createToolUsageDataPoint('2025-12-22T10:00:00Z', 'tool_a', 100),
        createToolUsageDataPoint('2025-12-22T11:00:00Z', 'tool_a', 150),
        createToolUsageDataPoint('2025-12-22T12:00:00Z', 'tool_a', 200),
      ]);

      const { result } = renderHook(() => useToolUsageChartData(defaultProps), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.chartData).toHaveLength(3);
      expect(result.current.chartData[0]).toHaveProperty('timestamp');
      expect(result.current.chartData[0]).toHaveProperty('tool_a', 100);
      expect(result.current.chartData[1]).toHaveProperty('tool_a', 150);
      expect(result.current.chartData[2]).toHaveProperty('tool_a', 200);
    });

    it('should fill missing time buckets with zeros', async () => {
      setupTraceMetricsHandler([
        // Only provide data for the first bucket
        createToolUsageDataPoint('2025-12-22T10:00:00Z', 'tool_a', 100),
      ]);

      const { result } = renderHook(() => useToolUsageChartData(defaultProps), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Should have all 3 time buckets
      expect(result.current.chartData).toHaveLength(3);
      // First bucket has data
      expect(result.current.chartData[0]).toHaveProperty('tool_a', 100);
      // Missing buckets should be filled with 0
      expect(result.current.chartData[1]).toHaveProperty('tool_a', 0);
      expect(result.current.chartData[2]).toHaveProperty('tool_a', 0);
    });

    it('should handle multiple tools with different data availability', async () => {
      setupTraceMetricsHandler([
        // Tool A has data in all buckets
        createToolUsageDataPoint('2025-12-22T10:00:00Z', 'tool_a', 100),
        createToolUsageDataPoint('2025-12-22T11:00:00Z', 'tool_a', 150),
        createToolUsageDataPoint('2025-12-22T12:00:00Z', 'tool_a', 200),
        // Tool B only has data in first bucket
        createToolUsageDataPoint('2025-12-22T10:00:00Z', 'tool_b', 50),
      ]);

      const { result } = renderHook(() => useToolUsageChartData(defaultProps), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.toolNames).toEqual(['tool_a', 'tool_b']);
      expect(result.current.chartData[0]).toHaveProperty('tool_a', 100);
      expect(result.current.chartData[0]).toHaveProperty('tool_b', 50);
      expect(result.current.chartData[1]).toHaveProperty('tool_a', 150);
      expect(result.current.chartData[1]).toHaveProperty('tool_b', 0);
      expect(result.current.chartData[2]).toHaveProperty('tool_a', 200);
      expect(result.current.chartData[2]).toHaveProperty('tool_b', 0);
    });

    it('should skip data points with missing tool name', async () => {
      setupTraceMetricsHandler([
        {
          metric_name: SpanMetricKey.SPAN_COUNT,
          dimensions: {
            time_bucket: '2025-12-22T10:00:00Z',
            // Missing SPAN_NAME
          },
          values: { [AggregationType.COUNT]: 100 },
        },
        createToolUsageDataPoint('2025-12-22T10:00:00Z', 'valid_tool', 50),
      ]);

      const { result } = renderHook(() => useToolUsageChartData(defaultProps), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Only valid_tool should be included
      expect(result.current.toolNames).toEqual(['valid_tool']);
    });

    it('should skip data points with missing time_bucket', async () => {
      setupTraceMetricsHandler([
        {
          metric_name: SpanMetricKey.SPAN_COUNT,
          dimensions: {
            [SpanDimensionKey.SPAN_NAME]: 'orphan_tool',
            // Missing time_bucket
          },
          values: { [AggregationType.COUNT]: 100 },
        },
        createToolUsageDataPoint('2025-12-22T10:00:00Z', 'valid_tool', 50),
      ]);

      const { result } = renderHook(() => useToolUsageChartData(defaultProps), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // orphan_tool should still be in toolNames (collected from data points)
      // but valid_tool should have data
      expect(result.current.toolNames).toContain('valid_tool');
    });

    it('should handle data points with missing count value', async () => {
      setupTraceMetricsHandler([
        {
          metric_name: SpanMetricKey.SPAN_COUNT,
          dimensions: {
            time_bucket: '2025-12-22T10:00:00Z',
            [SpanDimensionKey.SPAN_NAME]: 'test_tool',
          },
          values: {}, // Missing COUNT value
        },
      ]);

      const { result } = renderHook(() => useToolUsageChartData(defaultProps), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Tool should still be tracked, value defaults to 0
      expect(result.current.toolNames).toEqual(['test_tool']);
      expect(result.current.chartData[0]).toHaveProperty('test_tool', 0);
    });

    it('should return hasData true when there are tool names', async () => {
      setupTraceMetricsHandler([createToolUsageDataPoint('2025-12-22T10:00:00Z', 'tool_a', 100)]);

      const { result } = renderHook(() => useToolUsageChartData(defaultProps), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.hasData).toBe(true);
    });
  });

  describe('API request', () => {
    it('should include SPAN_NAME dimension in request', async () => {
      let capturedBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useToolUsageChartData(defaultProps), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(capturedBody).not.toBeNull();
      });

      expect(capturedBody.dimensions).toContain(SpanDimensionKey.SPAN_NAME);
    });

    it('should filter for TOOL type spans', async () => {
      let capturedBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useToolUsageChartData(defaultProps), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(capturedBody).not.toBeNull();
      });

      expect(capturedBody.filters).toContainEqual('span.type = "TOOL"');
    });
  });
});
