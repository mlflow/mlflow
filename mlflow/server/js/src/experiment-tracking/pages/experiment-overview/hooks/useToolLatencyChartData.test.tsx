import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useToolLatencyChartData } from './useToolLatencyChartData';
import { AggregationType, SpanMetricKey, SpanDimensionKey } from '@databricks/web-shared/model-trace-explorer';
import type { ReactNode } from 'react';

// Mock FetchUtils
jest.mock('../../../../common/utils/FetchUtils', () => ({
  fetchOrFail: jest.fn(),
  getAjaxUrl: (url: string) => url,
}));

import { fetchOrFail } from '../../../../common/utils/FetchUtils';
const mockFetchOrFail = fetchOrFail as jest.MockedFunction<typeof fetchOrFail>;

// Helper to create mock API response
const mockApiResponse = (dataPoints: any[] | undefined) => {
  mockFetchOrFail.mockResolvedValue({
    json: () => Promise.resolve({ data_points: dataPoints }),
  } as Response);
};

// Helper to create a tool latency data point
const createToolLatencyDataPoint = (timeBucket: string, toolName: string, avgLatency: number) => ({
  metric_name: SpanMetricKey.LATENCY,
  dimensions: {
    time_bucket: timeBucket,
    [SpanDimensionKey.SPAN_NAME]: toolName,
  },
  values: { [AggregationType.AVG]: avgLatency },
});

describe('useToolLatencyChartData', () => {
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

  beforeEach(() => {
    jest.clearAllMocks();
    mockApiResponse([]);
  });

  describe('loading state', () => {
    it('should return isLoading true while fetching', async () => {
      mockFetchOrFail.mockReturnValue(new Promise(() => {})); // Never resolve

      const { result } = renderHook(() => useToolLatencyChartData(defaultProps), {
        wrapper: createWrapper(),
      });

      expect(result.current.isLoading).toBe(true);
      expect(result.current.toolNames).toEqual([]);
    });
  });

  describe('error state', () => {
    it('should return error when API call fails', async () => {
      // Suppress expected console.error from react-query
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

      mockFetchOrFail.mockRejectedValue(new Error('API Error'));

      const { result } = renderHook(() => useToolLatencyChartData(defaultProps), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.error).toBeTruthy();
      });

      consoleSpy.mockRestore();
    });
  });

  describe('empty data', () => {
    it('should return hasData false when no data points', async () => {
      mockApiResponse([]);

      const { result } = renderHook(() => useToolLatencyChartData(defaultProps), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.hasData).toBe(false);
      expect(result.current.chartData).toHaveLength(3);
      expect(result.current.toolNames).toEqual([]);
    });

    it('should return hasData false when time range is not provided', async () => {
      mockApiResponse([]);

      const { result } = renderHook(
        () =>
          useToolLatencyChartData({
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
      mockApiResponse([
        createToolLatencyDataPoint('2025-12-22T10:00:00Z', 'zebra_tool', 100),
        createToolLatencyDataPoint('2025-12-22T10:00:00Z', 'alpha_tool', 200),
        createToolLatencyDataPoint('2025-12-22T11:00:00Z', 'beta_tool', 150),
      ]);

      const { result } = renderHook(() => useToolLatencyChartData(defaultProps), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.toolNames).toEqual(['alpha_tool', 'beta_tool', 'zebra_tool']);
    });

    it('should create chart data with all time buckets', async () => {
      mockApiResponse([
        createToolLatencyDataPoint('2025-12-22T10:00:00Z', 'tool_a', 100),
        createToolLatencyDataPoint('2025-12-22T11:00:00Z', 'tool_a', 150),
        createToolLatencyDataPoint('2025-12-22T12:00:00Z', 'tool_a', 200),
      ]);

      const { result } = renderHook(() => useToolLatencyChartData(defaultProps), {
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
      mockApiResponse([
        // Only provide data for the first bucket
        createToolLatencyDataPoint('2025-12-22T10:00:00Z', 'tool_a', 100),
      ]);

      const { result } = renderHook(() => useToolLatencyChartData(defaultProps), {
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
      mockApiResponse([
        // Tool A has data in all buckets
        createToolLatencyDataPoint('2025-12-22T10:00:00Z', 'tool_a', 100),
        createToolLatencyDataPoint('2025-12-22T11:00:00Z', 'tool_a', 150),
        createToolLatencyDataPoint('2025-12-22T12:00:00Z', 'tool_a', 200),
        // Tool B only has data in first bucket
        createToolLatencyDataPoint('2025-12-22T10:00:00Z', 'tool_b', 50),
      ]);

      const { result } = renderHook(() => useToolLatencyChartData(defaultProps), {
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
      mockApiResponse([
        {
          metric_name: SpanMetricKey.LATENCY,
          dimensions: {
            time_bucket: '2025-12-22T10:00:00Z',
            // Missing SPAN_NAME
          },
          values: { [AggregationType.AVG]: 100 },
        },
        createToolLatencyDataPoint('2025-12-22T10:00:00Z', 'valid_tool', 50),
      ]);

      const { result } = renderHook(() => useToolLatencyChartData(defaultProps), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Only valid_tool should be included
      expect(result.current.toolNames).toEqual(['valid_tool']);
    });

    it('should return hasData true when there are tool names', async () => {
      mockApiResponse([createToolLatencyDataPoint('2025-12-22T10:00:00Z', 'tool_a', 100)]);

      const { result } = renderHook(() => useToolLatencyChartData(defaultProps), {
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
      mockApiResponse([]);

      renderHook(() => useToolLatencyChartData(defaultProps), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(mockFetchOrFail).toHaveBeenCalled();
      });

      const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
      expect(callBody.dimensions).toContain(SpanDimensionKey.SPAN_NAME);
    });

    it('should filter for TOOL type spans', async () => {
      mockApiResponse([]);

      renderHook(() => useToolLatencyChartData(defaultProps), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(mockFetchOrFail).toHaveBeenCalled();
      });

      const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
      expect(callBody.filters).toContainEqual('span.type = "TOOL"');
    });

    it('should request AVG aggregation for latency', async () => {
      mockApiResponse([]);

      renderHook(() => useToolLatencyChartData(defaultProps), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(mockFetchOrFail).toHaveBeenCalled();
      });

      const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
      expect(callBody.aggregations).toContainEqual({ aggregation_type: AggregationType.AVG });
    });
  });
});
