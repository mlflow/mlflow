import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useTraceCostOverTimeChartData } from './useTraceCostOverTimeChartData';
import {
  AggregationType,
  SpanMetricKey,
  SpanDimensionKey,
  MetricViewType,
} from '@databricks/web-shared/model-trace-explorer';
import type { ReactNode } from 'react';
import { setupServer } from '../../../../common/utils/setup-msw';
import { rest } from 'msw';
import { OverviewChartProvider } from '../OverviewChartContext';

// Helper to create a cost data point with time bucket
const createCostDataPoint = (timeBucket: string, modelName: string, cost: number) => ({
  metric_name: SpanMetricKey.TOTAL_COST,
  dimensions: {
    time_bucket: timeBucket,
    [SpanDimensionKey.MODEL_NAME]: modelName,
  },
  values: { [AggregationType.SUM]: cost },
});

describe('useTraceCostOverTimeChartData', () => {
  const testExperimentId = 'test-experiment-123';
  const startTimeMs = new Date('2025-12-22T10:00:00Z').getTime();
  const endTimeMs = new Date('2025-12-22T12:00:00Z').getTime();
  const timeIntervalSeconds = 3600;

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

  const createWrapper = (contextOverrides: Partial<typeof contextProps> = {}) => {
    const queryClient = createQueryClient();
    const mergedContextProps = { ...contextProps, ...contextOverrides };
    return ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>
        <OverviewChartProvider {...mergedContextProps}>{children}</OverviewChartProvider>
      </QueryClientProvider>
    );
  };

  const setupTraceMetricsHandler = (dataPoints: any[]) => {
    server.use(
      rest.post('ajax-api/3.0/mlflow/traces/metrics', (_req, res, ctx) => {
        return res(ctx.json({ data_points: dataPoints }));
      }),
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
    setupTraceMetricsHandler([]);
  });

  describe('loading state', () => {
    it('should return isLoading true while fetching', async () => {
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', (_req, res, ctx) => {
          return res(ctx.delay('infinite'));
        }),
      );

      const { result } = renderHook(() => useTraceCostOverTimeChartData(), {
        wrapper: createWrapper(),
      });

      expect(result.current.isLoading).toBe(true);
      expect(result.current.chartData).toHaveLength(3); // Time buckets are pre-filled
      expect(result.current.modelNames).toHaveLength(0);
      expect(result.current.totalCost).toBe(0);
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

      const { result } = renderHook(() => useTraceCostOverTimeChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.error).toBeTruthy();
      });
    });
  });

  describe('empty data', () => {
    it('should return hasData false when no data points', async () => {
      const { result } = renderHook(() => useTraceCostOverTimeChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.hasData).toBe(false);
      expect(result.current.modelNames).toHaveLength(0);
      expect(result.current.totalCost).toBe(0);
    });
  });

  describe('data transformation', () => {
    it('should calculate total cost correctly', async () => {
      setupTraceMetricsHandler([
        createCostDataPoint('2025-12-22T10:00:00Z', 'gpt-4', 0.05),
        createCostDataPoint('2025-12-22T10:00:00Z', 'gpt-3.5-turbo', 0.03),
        createCostDataPoint('2025-12-22T11:00:00Z', 'gpt-4', 0.02),
      ]);

      const { result } = renderHook(() => useTraceCostOverTimeChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Total = 0.05 + 0.03 + 0.02 = 0.10
      expect(result.current.totalCost).toBeCloseTo(0.1, 10);
    });

    it('should extract unique model names sorted alphabetically', async () => {
      setupTraceMetricsHandler([
        createCostDataPoint('2025-12-22T10:00:00Z', 'gpt-4', 0.05),
        createCostDataPoint('2025-12-22T10:00:00Z', 'claude-3', 0.03),
        createCostDataPoint('2025-12-22T11:00:00Z', 'gpt-4', 0.02),
        createCostDataPoint('2025-12-22T11:00:00Z', 'gpt-3.5', 0.01),
      ]);

      const { result } = renderHook(() => useTraceCostOverTimeChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.modelNames).toEqual(['claude-3', 'gpt-3.5', 'gpt-4']);
    });

    it('should fill all time buckets with data', async () => {
      setupTraceMetricsHandler([
        createCostDataPoint('2025-12-22T10:00:00Z', 'gpt-4', 0.05),
        createCostDataPoint('2025-12-22T11:00:00Z', 'gpt-4', 0.03),
      ]);

      const { result } = renderHook(() => useTraceCostOverTimeChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Should have 3 time buckets
      expect(result.current.chartData).toHaveLength(3);
    });

    it('should fill missing time buckets with zero values', async () => {
      setupTraceMetricsHandler([
        createCostDataPoint('2025-12-22T10:00:00Z', 'gpt-4', 0.05),
        // No data for 11:00:00
        createCostDataPoint('2025-12-22T12:00:00Z', 'gpt-4', 0.03),
      ]);

      const { result } = renderHook(() => useTraceCostOverTimeChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Middle bucket should have 0 for gpt-4
      expect(result.current.chartData[1]['gpt-4']).toBe(0);
    });

    it('should fill missing models with zero values for each time bucket', async () => {
      setupTraceMetricsHandler([
        createCostDataPoint('2025-12-22T10:00:00Z', 'gpt-4', 0.05),
        createCostDataPoint('2025-12-22T10:00:00Z', 'claude-3', 0.03),
        createCostDataPoint('2025-12-22T11:00:00Z', 'gpt-4', 0.02),
        // claude-3 missing for 11:00:00
      ]);

      const { result } = renderHook(() => useTraceCostOverTimeChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // First bucket should have both models
      expect(result.current.chartData[0]['gpt-4']).toBe(0.05);
      expect(result.current.chartData[0]['claude-3']).toBe(0.03);

      // Second bucket should have gpt-4 and claude-3 (with 0)
      expect(result.current.chartData[1]['gpt-4']).toBe(0.02);
      expect(result.current.chartData[1]['claude-3']).toBe(0);
    });

    it('should handle single model correctly', async () => {
      setupTraceMetricsHandler([
        createCostDataPoint('2025-12-22T10:00:00Z', 'gpt-4', 0.05),
        createCostDataPoint('2025-12-22T11:00:00Z', 'gpt-4', 0.03),
        createCostDataPoint('2025-12-22T12:00:00Z', 'gpt-4', 0.02),
      ]);

      const { result } = renderHook(() => useTraceCostOverTimeChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.modelNames).toEqual(['gpt-4']);
      expect(result.current.chartData[0]['gpt-4']).toBe(0.05);
      expect(result.current.chartData[1]['gpt-4']).toBe(0.03);
      expect(result.current.chartData[2]['gpt-4']).toBe(0.02);
      expect(result.current.totalCost).toBeCloseTo(0.1, 10);
    });

    it('should handle multiple models at same time bucket', async () => {
      setupTraceMetricsHandler([
        createCostDataPoint('2025-12-22T10:00:00Z', 'gpt-4', 0.05),
        createCostDataPoint('2025-12-22T10:00:00Z', 'claude-3', 0.03),
        createCostDataPoint('2025-12-22T10:00:00Z', 'gpt-3.5', 0.02),
      ]);

      const { result } = renderHook(() => useTraceCostOverTimeChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.chartData[0]['gpt-4']).toBe(0.05);
      expect(result.current.chartData[0]['claude-3']).toBe(0.03);
      expect(result.current.chartData[0]['gpt-3.5']).toBe(0.02);
    });

    it('should handle missing cost values as zero', async () => {
      setupTraceMetricsHandler([
        createCostDataPoint('2025-12-22T10:00:00Z', 'gpt-4', 0.05),
        {
          metric_name: SpanMetricKey.TOTAL_COST,
          dimensions: {
            time_bucket: '2025-12-22T10:00:00Z',
            [SpanDimensionKey.MODEL_NAME]: 'claude-3',
          },
          values: {}, // Missing SUM value
        },
      ]);

      const { result } = renderHook(() => useTraceCostOverTimeChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.chartData[0]['claude-3']).toBe(0);
    });

    it('should handle very small costs correctly', async () => {
      setupTraceMetricsHandler([
        createCostDataPoint('2025-12-22T10:00:00Z', 'gpt-4', 0.000001),
        createCostDataPoint('2025-12-22T11:00:00Z', 'gpt-4', 0.000002),
      ]);

      const { result } = renderHook(() => useTraceCostOverTimeChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.totalCost).toBeCloseTo(0.000003, 10);
      expect(result.current.chartData[0]['gpt-4']).toBe(0.000001);
      expect(result.current.chartData[1]['gpt-4']).toBe(0.000002);
    });

    it('should handle large costs correctly', async () => {
      setupTraceMetricsHandler([
        createCostDataPoint('2025-12-22T10:00:00Z', 'gpt-4', 1000.5),
        createCostDataPoint('2025-12-22T11:00:00Z', 'gpt-4', 500.25),
      ]);

      const { result } = renderHook(() => useTraceCostOverTimeChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.totalCost).toBeCloseTo(1500.75, 2);
      expect(result.current.chartData[0]['gpt-4']).toBe(1000.5);
      expect(result.current.chartData[1]['gpt-4']).toBe(500.25);
    });
  });

  describe('API request', () => {
    it('should request SUM aggregation', async () => {
      let capturedBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useTraceCostOverTimeChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(capturedBody).not.toBeNull();
      });

      expect(capturedBody.aggregations).toContainEqual({ aggregation_type: AggregationType.SUM });
    });

    it('should request MODEL_NAME dimension', async () => {
      let capturedBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useTraceCostOverTimeChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(capturedBody).not.toBeNull();
      });

      expect(capturedBody.dimensions).toContain(SpanDimensionKey.MODEL_NAME);
    });

    it('should use SPANS view type', async () => {
      let capturedBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useTraceCostOverTimeChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(capturedBody).not.toBeNull();
      });

      expect(capturedBody.view_type).toBe(MetricViewType.SPANS);
    });

    it('should request TOTAL_COST metric', async () => {
      let capturedBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useTraceCostOverTimeChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(capturedBody).not.toBeNull();
      });

      expect(capturedBody.metric_name).toBe(SpanMetricKey.TOTAL_COST);
    });

    it('should include time interval in API call', async () => {
      let capturedBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useTraceCostOverTimeChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(capturedBody).not.toBeNull();
      });

      expect(capturedBody.time_interval_seconds).toBe(timeIntervalSeconds);
    });

    it('should include time range in API call', async () => {
      let capturedBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useTraceCostOverTimeChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(capturedBody).not.toBeNull();
      });

      expect(capturedBody.start_time_ms).toBe(startTimeMs);
      expect(capturedBody.end_time_ms).toBe(endTimeMs);
    });

    it('should include experiment ID in API call', async () => {
      let capturedBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useTraceCostOverTimeChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(capturedBody).not.toBeNull();
      });

      expect(capturedBody.experiment_ids).toContain(testExperimentId);
    });
  });
});
