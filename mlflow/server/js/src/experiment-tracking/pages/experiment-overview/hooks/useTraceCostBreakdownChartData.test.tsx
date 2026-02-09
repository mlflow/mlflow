import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useTraceCostBreakdownChartData } from './useTraceCostBreakdownChartData';
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

// Helper to create a cost breakdown data point
const createCostDataPoint = (modelName: string, totalCost: number) => ({
  metric_name: SpanMetricKey.TOTAL_COST,
  dimensions: { [SpanDimensionKey.MODEL_NAME]: modelName },
  values: { [AggregationType.SUM]: totalCost },
});

describe('useTraceCostBreakdownChartData', () => {
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

      const { result } = renderHook(() => useTraceCostBreakdownChartData(), {
        wrapper: createWrapper(),
      });

      expect(result.current.isLoading).toBe(true);
      expect(result.current.chartData).toHaveLength(0);
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

      const { result } = renderHook(() => useTraceCostBreakdownChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.error).toBeTruthy();
      });
    });
  });

  describe('empty data', () => {
    it('should return hasData false when no data points', async () => {
      const { result } = renderHook(() => useTraceCostBreakdownChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.hasData).toBe(false);
      expect(result.current.chartData).toHaveLength(0);
      expect(result.current.totalCost).toBe(0);
    });

    it('should return hasData false when all costs are zero', async () => {
      setupTraceMetricsHandler([createCostDataPoint('gpt-4', 0), createCostDataPoint('gpt-3.5', 0)]);

      const { result } = renderHook(() => useTraceCostBreakdownChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.hasData).toBe(false);
      expect(result.current.chartData).toHaveLength(0);
    });
  });

  describe('data transformation', () => {
    it('should calculate total cost correctly', async () => {
      setupTraceMetricsHandler([
        createCostDataPoint('gpt-4', 0.05),
        createCostDataPoint('gpt-3.5-turbo', 0.03),
        createCostDataPoint('claude-3', 0.02),
      ]);

      const { result } = renderHook(() => useTraceCostBreakdownChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Total = 0.05 + 0.03 + 0.02 = 0.10
      expect(result.current.totalCost).toBeCloseTo(0.1, 10);
    });

    it('should calculate percentages correctly', async () => {
      setupTraceMetricsHandler([
        createCostDataPoint('gpt-4', 0.5), // 50%
        createCostDataPoint('gpt-3.5', 0.3), // 30%
        createCostDataPoint('claude-3', 0.2), // 20%
      ]);

      const { result } = renderHook(() => useTraceCostBreakdownChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.chartData).toHaveLength(3);
      expect(result.current.chartData[0].percentage).toBeCloseTo(50, 1);
      expect(result.current.chartData[1].percentage).toBeCloseTo(30, 1);
      expect(result.current.chartData[2].percentage).toBeCloseTo(20, 1);
    });

    it('should sort models by cost descending', async () => {
      setupTraceMetricsHandler([
        createCostDataPoint('cheap-model', 0.01),
        createCostDataPoint('expensive-model', 0.1),
        createCostDataPoint('medium-model', 0.05),
      ]);

      const { result } = renderHook(() => useTraceCostBreakdownChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.chartData[0].name).toBe('expensive-model');
      expect(result.current.chartData[1].name).toBe('medium-model');
      expect(result.current.chartData[2].name).toBe('cheap-model');
    });

    it('should filter out models with zero cost', async () => {
      setupTraceMetricsHandler([createCostDataPoint('active-model', 0.05), createCostDataPoint('zero-cost-model', 0)]);

      const { result } = renderHook(() => useTraceCostBreakdownChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.chartData).toHaveLength(1);
      expect(result.current.chartData[0].name).toBe('active-model');
    });

    it('should use "Unknown" for missing model name', async () => {
      setupTraceMetricsHandler([
        {
          metric_name: SpanMetricKey.TOTAL_COST,
          dimensions: {}, // Missing model name
          values: { [AggregationType.SUM]: 0.05 },
        },
      ]);

      const { result } = renderHook(() => useTraceCostBreakdownChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.chartData[0].name).toBe('Unknown');
    });

    it('should handle missing values as zero cost', async () => {
      setupTraceMetricsHandler([
        {
          metric_name: SpanMetricKey.TOTAL_COST,
          dimensions: { [SpanDimensionKey.MODEL_NAME]: 'gpt-4' },
          values: {}, // Missing SUM value
        },
      ]);

      const { result } = renderHook(() => useTraceCostBreakdownChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Zero cost items are filtered out
      expect(result.current.chartData).toHaveLength(0);
      expect(result.current.hasData).toBe(false);
    });

    it('should return correct data structure for each model', async () => {
      setupTraceMetricsHandler([createCostDataPoint('gpt-4', 0.1)]);

      const { result } = renderHook(() => useTraceCostBreakdownChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      const dataPoint = result.current.chartData[0];
      expect(dataPoint).toHaveProperty('name', 'gpt-4');
      expect(dataPoint).toHaveProperty('value', 0.1);
      expect(dataPoint).toHaveProperty('percentage', 100);
    });

    it('should handle single model with 100% percentage', async () => {
      setupTraceMetricsHandler([createCostDataPoint('only-model', 1.23)]);

      const { result } = renderHook(() => useTraceCostBreakdownChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.chartData).toHaveLength(1);
      expect(result.current.chartData[0].percentage).toBe(100);
      expect(result.current.totalCost).toBe(1.23);
    });

    it('should handle very small costs correctly', async () => {
      setupTraceMetricsHandler([createCostDataPoint('gpt-4', 0.000001), createCostDataPoint('gpt-3.5', 0.000002)]);

      const { result } = renderHook(() => useTraceCostBreakdownChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.totalCost).toBeCloseTo(0.000003, 10);
      expect(result.current.chartData[0].percentage).toBeCloseTo(66.67, 1);
      expect(result.current.chartData[1].percentage).toBeCloseTo(33.33, 1);
    });

    it('should handle large costs correctly', async () => {
      setupTraceMetricsHandler([createCostDataPoint('gpt-4', 1000.5), createCostDataPoint('gpt-3.5', 500.25)]);

      const { result } = renderHook(() => useTraceCostBreakdownChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.totalCost).toBeCloseTo(1500.75, 2);
      expect(result.current.chartData[0].name).toBe('gpt-4');
      expect(result.current.chartData[0].value).toBe(1000.5);
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

      renderHook(() => useTraceCostBreakdownChartData(), {
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

      renderHook(() => useTraceCostBreakdownChartData(), {
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

      renderHook(() => useTraceCostBreakdownChartData(), {
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

      renderHook(() => useTraceCostBreakdownChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(capturedBody).not.toBeNull();
      });

      expect(capturedBody.metric_name).toBe(SpanMetricKey.TOTAL_COST);
    });

    it('should include time range in API call', async () => {
      let capturedBody: any = null;

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useTraceCostBreakdownChartData(), {
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

      renderHook(() => useTraceCostBreakdownChartData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(capturedBody).not.toBeNull();
      });

      expect(capturedBody.experiment_ids).toContain(testExperimentId);
    });
  });
});
