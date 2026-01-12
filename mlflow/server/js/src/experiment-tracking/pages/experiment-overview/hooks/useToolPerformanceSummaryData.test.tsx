import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useToolPerformanceSummaryData } from './useToolPerformanceSummaryData';
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

describe('useToolPerformanceSummaryData', () => {
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
      <QueryClientProvider client={queryClient}>
        <OverviewChartProvider {...contextProps}>{children}</OverviewChartProvider>
      </QueryClientProvider>
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
    it('should return isLoading true while fetching', async () => {
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', (_req, res, ctx) => {
          return res(ctx.delay('infinite'));
        }),
      );

      const { result } = renderHook(() => useToolPerformanceSummaryData(), {
        wrapper: createWrapper(),
      });

      expect(result.current.isLoading).toBe(true);
      expect(result.current.toolsData).toEqual([]);
    });
  });

  describe('error state', () => {
    it('should return error when API call fails', async () => {
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', (_req, res, ctx) => {
          return res(ctx.status(500), ctx.json({ error: 'API Error' }));
        }),
      );

      const { result } = renderHook(() => useToolPerformanceSummaryData(), {
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

      const { result } = renderHook(() => useToolPerformanceSummaryData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.hasData).toBe(false);
      expect(result.current.toolsData).toEqual([]);
    });
  });

  describe('data transformation', () => {
    it('should calculate success rate from OK and ERROR counts', async () => {
      setupTraceMetricsHandler(
        [createCountDataPoint('tool_a', SpanStatus.OK, 90), createCountDataPoint('tool_a', SpanStatus.ERROR, 10)],
        [createLatencyDataPoint('tool_a', 150)],
      );

      const { result } = renderHook(() => useToolPerformanceSummaryData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.toolsData).toHaveLength(1);
      expect(result.current.toolsData[0].toolName).toBe('tool_a');
      expect(result.current.toolsData[0].totalCalls).toBe(100);
      expect(result.current.toolsData[0].successRate).toBe(90); // 90%
      expect(result.current.toolsData[0].avgLatency).toBe(150);
    });

    it('should handle 100% success rate', async () => {
      setupTraceMetricsHandler(
        [createCountDataPoint('tool_a', SpanStatus.OK, 100)],
        [createLatencyDataPoint('tool_a', 200)],
      );

      const { result } = renderHook(() => useToolPerformanceSummaryData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.toolsData[0].successRate).toBe(100);
    });

    it('should handle 0% success rate', async () => {
      setupTraceMetricsHandler(
        [createCountDataPoint('tool_a', SpanStatus.ERROR, 50)],
        [createLatencyDataPoint('tool_a', 100)],
      );

      const { result } = renderHook(() => useToolPerformanceSummaryData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.toolsData[0].successRate).toBe(0);
      expect(result.current.toolsData[0].totalCalls).toBe(50);
    });

    it('should sort tools by total calls descending', async () => {
      setupTraceMetricsHandler(
        [
          createCountDataPoint('low_usage_tool', SpanStatus.OK, 10),
          createCountDataPoint('high_usage_tool', SpanStatus.OK, 1000),
          createCountDataPoint('medium_usage_tool', SpanStatus.OK, 100),
        ],
        [
          createLatencyDataPoint('low_usage_tool', 50),
          createLatencyDataPoint('high_usage_tool', 100),
          createLatencyDataPoint('medium_usage_tool', 75),
        ],
      );

      const { result } = renderHook(() => useToolPerformanceSummaryData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.toolsData).toHaveLength(3);
      expect(result.current.toolsData[0].toolName).toBe('high_usage_tool');
      expect(result.current.toolsData[1].toolName).toBe('medium_usage_tool');
      expect(result.current.toolsData[2].toolName).toBe('low_usage_tool');
    });

    it('should handle multiple tools with different statuses', async () => {
      setupTraceMetricsHandler(
        [
          createCountDataPoint('tool_a', SpanStatus.OK, 80),
          createCountDataPoint('tool_a', SpanStatus.ERROR, 20),
          createCountDataPoint('tool_b', SpanStatus.OK, 45),
          createCountDataPoint('tool_b', SpanStatus.ERROR, 5),
        ],
        [createLatencyDataPoint('tool_a', 100), createLatencyDataPoint('tool_b', 200)],
      );

      const { result } = renderHook(() => useToolPerformanceSummaryData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.toolsData).toHaveLength(2);
      // tool_a has 100 total calls, tool_b has 50
      expect(result.current.toolsData[0].toolName).toBe('tool_a');
      expect(result.current.toolsData[0].totalCalls).toBe(100);
      expect(result.current.toolsData[0].successRate).toBe(80);
      expect(result.current.toolsData[1].toolName).toBe('tool_b');
      expect(result.current.toolsData[1].totalCalls).toBe(50);
      expect(result.current.toolsData[1].successRate).toBe(90);
    });

    it('should default latency to 0 if not found', async () => {
      setupTraceMetricsHandler(
        [createCountDataPoint('tool_a', SpanStatus.OK, 100)],
        [], // No latency data
      );

      const { result } = renderHook(() => useToolPerformanceSummaryData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.toolsData[0].avgLatency).toBe(0);
    });

    it('should return hasData true when there are tools', async () => {
      setupTraceMetricsHandler(
        [createCountDataPoint('tool_a', SpanStatus.OK, 100)],
        [createLatencyDataPoint('tool_a', 50)],
      );

      const { result } = renderHook(() => useToolPerformanceSummaryData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.hasData).toBe(true);
    });
  });

  describe('API request', () => {
    it('should include SPAN_NAME and SPAN_STATUS dimensions in count request', async () => {
      const capturedBodies: any[] = [];

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBodies.push(await req.json());
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useToolPerformanceSummaryData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(capturedBodies.length).toBeGreaterThanOrEqual(2);
      });

      const countRequest = capturedBodies.find((b) => b.metric_name === SpanMetricKey.SPAN_COUNT);
      expect(countRequest.dimensions).toContain(SpanDimensionKey.SPAN_NAME);
      expect(countRequest.dimensions).toContain(SpanDimensionKey.SPAN_STATUS);
    });

    it('should include SPAN_NAME dimension in latency request', async () => {
      const capturedBodies: any[] = [];

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBodies.push(await req.json());
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useToolPerformanceSummaryData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(capturedBodies.length).toBeGreaterThanOrEqual(2);
      });

      const latencyRequest = capturedBodies.find((b) => b.metric_name === SpanMetricKey.LATENCY);
      expect(latencyRequest.dimensions).toContain(SpanDimensionKey.SPAN_NAME);
    });

    it('should filter for TOOL type spans', async () => {
      const capturedBodies: any[] = [];

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBodies.push(await req.json());
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useToolPerformanceSummaryData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(capturedBodies.length).toBeGreaterThanOrEqual(2);
      });

      // Both requests should filter for TOOL type
      for (const body of capturedBodies) {
        expect(body.filters).toContainEqual('span.type = "TOOL"');
      }
    });

    it('should NOT include time_interval_seconds (aggregate query)', async () => {
      const capturedBodies: any[] = [];

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBodies.push(await req.json());
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useToolPerformanceSummaryData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(capturedBodies.length).toBeGreaterThanOrEqual(2);
      });

      // Neither request should include time_interval_seconds
      for (const body of capturedBodies) {
        expect(body.time_interval_seconds).toBeUndefined();
      }
    });
  });
});
