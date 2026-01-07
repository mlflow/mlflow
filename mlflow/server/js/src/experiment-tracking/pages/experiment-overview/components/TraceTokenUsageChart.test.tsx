import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { TraceTokenUsageChart } from './TraceTokenUsageChart';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MetricViewType, AggregationType, TraceMetricKey } from '@databricks/web-shared/model-trace-explorer';

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

// Helper to chain mock API responses (for input tokens, output tokens, total tokens)
const mockApiResponses = (inputDataPoints: any[], outputDataPoints: any[], totalDataPoints: any[]) => {
  mockFetchOrFail
    .mockResolvedValueOnce({
      json: () => Promise.resolve({ data_points: inputDataPoints }),
    } as Response)
    .mockResolvedValueOnce({
      json: () => Promise.resolve({ data_points: outputDataPoints }),
    } as Response)
    .mockResolvedValueOnce({
      json: () => Promise.resolve({ data_points: totalDataPoints }),
    } as Response);
};

// Helper to create an input tokens data point
const createInputTokensDataPoint = (timeBucket: string, sum: number) => ({
  metric_name: TraceMetricKey.INPUT_TOKENS,
  dimensions: { time_bucket: timeBucket },
  values: { [AggregationType.SUM]: sum },
});

// Helper to create an output tokens data point
const createOutputTokensDataPoint = (timeBucket: string, sum: number) => ({
  metric_name: TraceMetricKey.OUTPUT_TOKENS,
  dimensions: { time_bucket: timeBucket },
  values: { [AggregationType.SUM]: sum },
});

// Helper to create a total tokens data point (no time bucket)
const createTotalTokensDataPoint = (sum: number) => ({
  metric_name: TraceMetricKey.TOTAL_TOKENS,
  dimensions: {},
  values: { [AggregationType.SUM]: sum },
});

describe('TraceTokenUsageChart', () => {
  const testExperimentId = 'test-experiment-123';
  // Use fixed timestamps for predictable bucket generation
  const startTimeMs = new Date('2025-12-22T10:00:00Z').getTime();
  const endTimeMs = new Date('2025-12-22T12:00:00Z').getTime(); // 2 hours = 3 buckets with 1hr interval
  const timeIntervalSeconds = 3600; // 1 hour

  // Pre-computed time buckets for the test range
  const timeBuckets = [
    new Date('2025-12-22T10:00:00Z').getTime(),
    new Date('2025-12-22T11:00:00Z').getTime(),
    new Date('2025-12-22T12:00:00Z').getTime(),
  ];

  // Default props reused across tests
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

  const renderComponent = (props: Partial<typeof defaultProps> = {}) => {
    const queryClient = createQueryClient();
    return renderWithIntl(
      <QueryClientProvider client={queryClient}>
        <DesignSystemProvider>
          <TraceTokenUsageChart {...defaultProps} {...props} />
        </DesignSystemProvider>
      </QueryClientProvider>,
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockApiResponse([]);
  });

  describe('loading state', () => {
    it('should render loading spinner while data is being fetched', async () => {
      // Create a promise that never resolves to keep the component in loading state
      mockFetchOrFail.mockReturnValue(new Promise(() => {}));

      renderComponent();

      // Check for spinner (loading state)
      expect(screen.getByRole('img')).toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('should render error message when API call fails', async () => {
      mockFetchOrFail.mockRejectedValue(new Error('API Error'));

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Failed to load chart data')).toBeInTheDocument();
      });
    });
  });

  describe('empty data state', () => {
    it('should render empty state when no data points are returned', async () => {
      mockApiResponses([], [], []);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('No data available for the selected time range')).toBeInTheDocument();
      });
    });

    it('should render empty state when time range is not provided', async () => {
      mockApiResponse([]);

      renderComponent({ startTimeMs: undefined, endTimeMs: undefined, timeBuckets: [] });

      await waitFor(() => {
        expect(screen.getByText('No data available for the selected time range')).toBeInTheDocument();
      });
    });
  });

  describe('with data', () => {
    const mockInputDataPoints = [
      createInputTokensDataPoint('2025-12-22T10:00:00Z', 50000),
      createInputTokensDataPoint('2025-12-22T11:00:00Z', 75000),
    ];

    const mockOutputDataPoints = [
      createOutputTokensDataPoint('2025-12-22T10:00:00Z', 20000),
      createOutputTokensDataPoint('2025-12-22T11:00:00Z', 30000),
    ];

    const mockTotalDataPoints = [createTotalTokensDataPoint(175000)];

    it('should render chart with data points', async () => {
      mockApiResponses(mockInputDataPoints, mockOutputDataPoints, mockTotalDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('area-chart')).toBeInTheDocument();
      });

      // Verify the area chart has all time buckets (3 buckets for 2-hour range with 1hr interval)
      expect(screen.getByTestId('area-chart')).toHaveAttribute('data-count', '3');
    });

    it('should display both input and output token areas', async () => {
      mockApiResponses(mockInputDataPoints, mockOutputDataPoints, mockTotalDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('area-inputTokens')).toBeInTheDocument();
        expect(screen.getByTestId('area-outputTokens')).toBeInTheDocument();
      });
    });

    it('should display the total tokens in header', async () => {
      mockApiResponses(mockInputDataPoints, mockOutputDataPoints, mockTotalDataPoints);

      renderComponent();

      // 175000 should be displayed as 175.00K
      await waitFor(() => {
        expect(screen.getByText('175.00K')).toBeInTheDocument();
      });
    });

    it('should display the "Token Usage" title', async () => {
      mockApiResponses(mockInputDataPoints, mockOutputDataPoints, mockTotalDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Token Usage')).toBeInTheDocument();
      });
    });

    it('should display input and output tokens in subtitle', async () => {
      mockApiResponses(mockInputDataPoints, mockOutputDataPoints, mockTotalDataPoints);

      renderComponent();

      // Subtitle shows input/output breakdown: (125.00K input, 50.00K output)
      // Input: 50000 + 75000 = 125000, Output: 20000 + 30000 = 50000
      await waitFor(() => {
        expect(screen.getByText(/125\.00K input/)).toBeInTheDocument();
        expect(screen.getByText(/50\.00K output/)).toBeInTheDocument();
      });
    });

    it('should display "Over time" label', async () => {
      mockApiResponses(mockInputDataPoints, mockOutputDataPoints, mockTotalDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Over time')).toBeInTheDocument();
      });
    });

    it('should format token count in millions for values >= 1,000,000', async () => {
      mockApiResponses(mockInputDataPoints, mockOutputDataPoints, [createTotalTokensDataPoint(1500000)]);

      renderComponent();

      // 1,500,000 should be displayed as 1.50M
      await waitFor(() => {
        expect(screen.getByText('1.50M')).toBeInTheDocument();
      });
    });

    it('should format token count in thousands for values >= 1,000', async () => {
      mockApiResponses(mockInputDataPoints, mockOutputDataPoints, [createTotalTokensDataPoint(5000)]);

      renderComponent();

      // 5000 should be displayed as 5.00K
      await waitFor(() => {
        expect(screen.getByText('5.00K')).toBeInTheDocument();
      });
    });

    it('should format token count with locale string for values < 1,000', async () => {
      mockApiResponses(mockInputDataPoints, mockOutputDataPoints, [createTotalTokensDataPoint(500)]);

      renderComponent();

      // 500 should be displayed as "500"
      await waitFor(() => {
        expect(screen.getByText('500')).toBeInTheDocument();
      });
    });
  });

  describe('API call parameters', () => {
    it('should call fetchOrFail for input tokens with correct parameters', async () => {
      mockApiResponses([], [], []);

      renderComponent();

      await waitFor(() => {
        const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
        expect(callBody).toMatchObject({
          experiment_ids: [testExperimentId],
          view_type: MetricViewType.TRACES,
          metric_name: TraceMetricKey.INPUT_TOKENS,
          aggregations: [{ aggregation_type: AggregationType.SUM }],
          time_interval_seconds: 3600,
        });
      });
    });

    it('should call fetchOrFail for output tokens with correct parameters', async () => {
      mockApiResponses([], [], []);

      renderComponent();

      await waitFor(() => {
        const callBody = JSON.parse((mockFetchOrFail.mock.calls[1]?.[1] as any)?.body || '{}');
        expect(callBody).toMatchObject({
          experiment_ids: [testExperimentId],
          view_type: MetricViewType.TRACES,
          metric_name: TraceMetricKey.OUTPUT_TOKENS,
          aggregations: [{ aggregation_type: AggregationType.SUM }],
          time_interval_seconds: 3600,
        });
      });
    });

    it('should call fetchOrFail for total tokens without time interval', async () => {
      mockApiResponses([], [], []);

      renderComponent();

      await waitFor(() => {
        const callBody = JSON.parse((mockFetchOrFail.mock.calls[2]?.[1] as any)?.body || '{}');
        expect(callBody).toMatchObject({
          experiment_ids: [testExperimentId],
          view_type: MetricViewType.TRACES,
          metric_name: TraceMetricKey.TOTAL_TOKENS,
          aggregations: [{ aggregation_type: AggregationType.SUM }],
        });
        // Should NOT have time_interval_seconds for total tokens query
        expect(callBody.time_interval_seconds).toBeUndefined();
      });
    });

    it('should use provided time interval', async () => {
      mockApiResponses([], [], []);

      renderComponent({ timeIntervalSeconds: 60 });

      await waitFor(() => {
        const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
        expect(callBody.time_interval_seconds).toBe(60);
      });
    });
  });

  describe('data transformation', () => {
    it('should handle data points with missing token values gracefully', async () => {
      mockApiResponses(
        [
          {
            metric_name: TraceMetricKey.INPUT_TOKENS,
            dimensions: { time_bucket: '2025-12-22T10:00:00Z' },
            values: {}, // Missing SUM value - will be treated as 0
          },
        ],
        [
          {
            metric_name: TraceMetricKey.OUTPUT_TOKENS,
            dimensions: { time_bucket: '2025-12-22T10:00:00Z' },
            values: {}, // Missing SUM value - will be treated as 0
          },
        ],
        [createTotalTokensDataPoint(0)],
      );

      renderComponent();

      // Should still render with all 3 time buckets
      await waitFor(() => {
        expect(screen.getByTestId('area-chart')).toHaveAttribute('data-count', '3');
      });
    });

    it('should handle missing total tokens data gracefully', async () => {
      mockApiResponses(
        [createInputTokensDataPoint('2025-12-22T10:00:00Z', 1000)],
        [createOutputTokensDataPoint('2025-12-22T10:00:00Z', 500)],
        [],
      );

      renderComponent();

      // Should still render the chart with all time buckets
      await waitFor(() => {
        expect(screen.getByTestId('area-chart')).toHaveAttribute('data-count', '3');
      });

      // Should display 0 when total tokens is not available
      expect(screen.getByText('0')).toBeInTheDocument();
    });

    it('should handle data points with missing time_bucket', async () => {
      mockApiResponses(
        [
          {
            metric_name: TraceMetricKey.INPUT_TOKENS,
            dimensions: {}, // Missing time_bucket - won't be mapped to any bucket
            values: { [AggregationType.SUM]: 1000 },
          },
        ],
        [
          {
            metric_name: TraceMetricKey.OUTPUT_TOKENS,
            dimensions: {}, // Missing time_bucket - won't be mapped to any bucket
            values: { [AggregationType.SUM]: 500 },
          },
        ],
        [createTotalTokensDataPoint(1500)],
      );

      renderComponent();

      // Should still render the chart with all generated time buckets (all with 0 values)
      await waitFor(() => {
        expect(screen.getByTestId('area-chart')).toHaveAttribute('data-count', '3');
      });
    });

    it('should fill missing time buckets with zeros', async () => {
      // Only provide data for one time bucket - the chart should still show all 3 buckets
      const inputDataPoints = [createInputTokensDataPoint('2025-12-22T10:00:00Z', 1000)];
      const outputDataPoints = [createOutputTokensDataPoint('2025-12-22T10:00:00Z', 500)];

      mockApiResponses(inputDataPoints, outputDataPoints, [createTotalTokensDataPoint(1500)]);

      renderComponent();

      // Should render chart with all 3 time buckets (missing ones filled with 0)
      await waitFor(() => {
        expect(screen.getByTestId('area-chart')).toHaveAttribute('data-count', '3');
      });
    });

    it('should merge input and output tokens by time bucket', async () => {
      // Input tokens has data for both time buckets
      const inputDataPoints = [
        createInputTokensDataPoint('2025-12-22T10:00:00Z', 1000),
        createInputTokensDataPoint('2025-12-22T11:00:00Z', 2000),
      ];

      // Output tokens only has data for the first time bucket (second bucket will be 0)
      const outputDataPoints = [createOutputTokensDataPoint('2025-12-22T10:00:00Z', 500)];

      mockApiResponses(inputDataPoints, outputDataPoints, [createTotalTokensDataPoint(3500)]);

      renderComponent();

      // Should render chart with all 3 time buckets
      await waitFor(() => {
        expect(screen.getByTestId('area-chart')).toHaveAttribute('data-count', '3');
      });
    });
  });
});
