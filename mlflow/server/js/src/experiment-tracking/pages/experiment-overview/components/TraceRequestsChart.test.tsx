import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { TraceRequestsChart } from './TraceRequestsChart';
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

// Helper to create a single data point
const createTraceCountDataPoint = (timeBucket: string, count: number) => ({
  metric_name: TraceMetricKey.TRACE_COUNT,
  dimensions: { time_bucket: timeBucket },
  values: { [AggregationType.COUNT]: count },
});

describe('TraceRequestsChart', () => {
  const testExperimentId = 'test-experiment-123';
  // Use fixed timestamps for predictable bucket generation
  const startTimeMs = new Date('2025-12-22T10:00:00Z').getTime();
  const endTimeMs = new Date('2025-12-22T12:00:00Z').getTime(); // 2 hours = 3 buckets with 1hr interval

  // Default props reused across tests
  const defaultProps = {
    experimentId: testExperimentId,
    startTimeMs,
    endTimeMs,
    timeIntervalSeconds: 3600, // 1 hour
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
          <TraceRequestsChart {...defaultProps} {...props} />
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
    it('should render chart with zeros when no data points are returned', async () => {
      mockApiResponse([]);

      renderComponent();

      // Chart should still render with all time buckets (filled with zeros)
      await waitFor(() => {
        expect(screen.getByTestId('bar-chart')).toHaveAttribute('data-count', '3');
      });

      // Total should be 0
      expect(screen.getByText('0')).toBeInTheDocument();
    });

    it('should render empty state when time range is not provided', async () => {
      mockApiResponse([]);

      renderComponent({ startTimeMs: undefined, endTimeMs: undefined });

      await waitFor(() => {
        expect(screen.getByText('No data available for the selected time range')).toBeInTheDocument();
      });
    });
  });

  describe('with data', () => {
    const mockDataPoints = [
      createTraceCountDataPoint('2025-12-22T10:00:00Z', 42),
      createTraceCountDataPoint('2025-12-22T11:00:00Z', 58),
      createTraceCountDataPoint('2025-12-22T12:00:00Z', 100),
    ];

    it('should render chart with all time buckets', async () => {
      mockApiResponse(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      });

      // Verify the bar chart has all 3 time buckets
      expect(screen.getByTestId('bar-chart')).toHaveAttribute('data-count', '3');
    });

    it('should display the total request count', async () => {
      mockApiResponse(mockDataPoints);

      renderComponent();

      // Total should be 42 + 58 + 100 = 200
      await waitFor(() => {
        expect(screen.getByText('200')).toBeInTheDocument();
      });
    });

    it('should display the "Requests" title', async () => {
      mockApiResponse(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Requests')).toBeInTheDocument();
      });
    });

    it('should format large numbers with locale formatting', async () => {
      mockApiResponse([createTraceCountDataPoint('2025-12-22T10:00:00Z', 1234567)]);

      renderComponent();

      await waitFor(() => {
        // Check for locale-formatted number (1,234,567 in US locale)
        expect(screen.getByText('1,234,567')).toBeInTheDocument();
      });
    });

    it('should fill missing time buckets with zeros', async () => {
      // Only provide data for one time bucket
      mockApiResponse([createTraceCountDataPoint('2025-12-22T10:00:00Z', 100)]);

      renderComponent();

      // Chart should still show all 3 time buckets
      await waitFor(() => {
        expect(screen.getByTestId('bar-chart')).toHaveAttribute('data-count', '3');
      });

      // Total should only count the one data point
      expect(screen.getByText('100')).toBeInTheDocument();
    });
  });

  describe('API call parameters', () => {
    it('should call fetchOrFail with correct parameters', async () => {
      renderComponent();

      await waitFor(() => {
        expect(mockFetchOrFail).toHaveBeenCalledWith(
          'ajax-api/3.0/mlflow/traces/metrics',
          expect.objectContaining({
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: expect.stringContaining(testExperimentId),
          }),
        );
      });
    });

    it('should use provided time interval', async () => {
      renderComponent({ timeIntervalSeconds: 60 });

      await waitFor(() => {
        const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
        expect(callBody.time_interval_seconds).toBe(60);
      });
    });

    it('should use provided time interval for hourly grouping', async () => {
      renderComponent({ timeIntervalSeconds: 3600 });

      await waitFor(() => {
        const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
        expect(callBody.time_interval_seconds).toBe(3600);
      });
    });

    it('should use provided time interval for daily grouping', async () => {
      renderComponent({ timeIntervalSeconds: 86400 });

      await waitFor(() => {
        const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
        expect(callBody.time_interval_seconds).toBe(86400);
      });
    });
  });

  describe('data transformation', () => {
    it('should handle data points with missing values gracefully', async () => {
      mockApiResponse([
        {
          metric_name: TraceMetricKey.TRACE_COUNT,
          dimensions: { time_bucket: '2025-12-22T10:00:00Z' },
          values: {}, // Missing COUNT value - will be treated as 0
        },
        createTraceCountDataPoint('2025-12-22T11:00:00Z', 50),
      ]);

      renderComponent();

      // Should still render with all time buckets and show total of 50 (0 + 50)
      await waitFor(() => {
        expect(screen.getByTestId('bar-chart')).toHaveAttribute('data-count', '3');
      });
      expect(screen.getByText('50')).toBeInTheDocument();
    });

    it('should handle data points with missing time_bucket', async () => {
      mockApiResponse([
        {
          metric_name: TraceMetricKey.TRACE_COUNT,
          dimensions: {}, // Missing time_bucket - won't be mapped to any bucket
          values: { [AggregationType.COUNT]: 25 },
        },
      ]);

      renderComponent();

      // Should still render with all time buckets (all with 0 values in chart, but total still counts)
      await waitFor(() => {
        expect(screen.getByTestId('bar-chart')).toHaveAttribute('data-count', '3');
      });
      // Total count still includes the data point
      expect(screen.getByText('25')).toBeInTheDocument();
    });
  });
});
