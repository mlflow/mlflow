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

// Mock recharts components to avoid rendering issues in tests
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="responsive-container">{children}</div>
  ),
  BarChart: ({ children, data }: { children: React.ReactNode; data: any[] }) => (
    <div data-testid="bar-chart" data-count={data?.length || 0}>
      {children}
    </div>
  ),
  Bar: () => <div data-testid="bar" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  Tooltip: () => <div data-testid="tooltip" />,
}));

describe('TraceRequestsChart', () => {
  const testExperimentId = 'test-experiment-123';
  const now = Date.now();
  const oneHourAgo = now - 60 * 60 * 1000;

  const createQueryClient = () =>
    new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

  const renderComponent = (props: { experimentId: string; startTimeMs?: number; endTimeMs?: number }) => {
    const queryClient = createQueryClient();
    return renderWithIntl(
      <QueryClientProvider client={queryClient}>
        <DesignSystemProvider>
          <TraceRequestsChart {...props} />
        </DesignSystemProvider>
      </QueryClientProvider>,
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
    // Mock fetchOrFail to return a response with json() method
    mockFetchOrFail.mockResolvedValue({
      json: () => Promise.resolve({ data_points: [] }),
    } as Response);
  });

  describe('loading state', () => {
    it('should render loading spinner while data is being fetched', async () => {
      // Create a promise that never resolves to keep the component in loading state
      mockFetchOrFail.mockReturnValue(new Promise(() => {}));

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs: oneHourAgo,
        endTimeMs: now,
      });

      // Check for spinner (loading state)
      expect(screen.getByRole('img')).toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('should render error message when API call fails', async () => {
      mockFetchOrFail.mockRejectedValue(new Error('API Error'));

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs: oneHourAgo,
        endTimeMs: now,
      });

      await waitFor(() => {
        expect(screen.getByText('Failed to load chart data')).toBeInTheDocument();
      });
    });
  });

  describe('empty data state', () => {
    it('should render empty state message when no data points are returned', async () => {
      mockFetchOrFail.mockResolvedValue({
        json: () => Promise.resolve({ data_points: [] }),
      } as Response);

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs: oneHourAgo,
        endTimeMs: now,
      });

      await waitFor(() => {
        expect(screen.getByText('No data available for the selected time range')).toBeInTheDocument();
      });
    });

    it('should render empty state when data_points is undefined', async () => {
      // Cast to simulate edge case where API might return malformed response
      mockFetchOrFail.mockResolvedValue({
        json: () => Promise.resolve({ data_points: undefined }),
      } as Response);

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs: oneHourAgo,
        endTimeMs: now,
      });

      await waitFor(() => {
        expect(screen.getByText('No data available for the selected time range')).toBeInTheDocument();
      });
    });
  });

  describe('with data', () => {
    const mockDataPoints = [
      {
        metric_name: TraceMetricKey.TRACE_COUNT,
        dimensions: { time_bucket: '2025-12-22T10:00:00Z' },
        values: { [AggregationType.COUNT]: 42 },
      },
      {
        metric_name: TraceMetricKey.TRACE_COUNT,
        dimensions: { time_bucket: '2025-12-22T11:00:00Z' },
        values: { [AggregationType.COUNT]: 58 },
      },
      {
        metric_name: TraceMetricKey.TRACE_COUNT,
        dimensions: { time_bucket: '2025-12-22T12:00:00Z' },
        values: { [AggregationType.COUNT]: 100 },
      },
    ];

    it('should render chart with data points', async () => {
      mockFetchOrFail.mockResolvedValue({
        json: () => Promise.resolve({ data_points: mockDataPoints }),
      } as Response);

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs: oneHourAgo,
        endTimeMs: now,
      });

      await waitFor(() => {
        expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      });

      // Verify the bar chart has the correct number of data points
      expect(screen.getByTestId('bar-chart')).toHaveAttribute('data-count', '3');
    });

    it('should display the total request count', async () => {
      mockFetchOrFail.mockResolvedValue({
        json: () => Promise.resolve({ data_points: mockDataPoints }),
      } as Response);

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs: oneHourAgo,
        endTimeMs: now,
      });

      // Total should be 42 + 58 + 100 = 200
      await waitFor(() => {
        expect(screen.getByText('200')).toBeInTheDocument();
      });
    });

    it('should display the "Requests" title', async () => {
      mockFetchOrFail.mockResolvedValue({
        json: () => Promise.resolve({ data_points: mockDataPoints }),
      } as Response);

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs: oneHourAgo,
        endTimeMs: now,
      });

      await waitFor(() => {
        expect(screen.getByText('Requests')).toBeInTheDocument();
      });
    });

    it('should format large numbers with locale formatting', async () => {
      mockFetchOrFail.mockResolvedValue({
        json: () =>
          Promise.resolve({
            data_points: [
              {
                metric_name: TraceMetricKey.TRACE_COUNT,
                dimensions: { time_bucket: '2025-12-22T10:00:00Z' },
                values: { [AggregationType.COUNT]: 1234567 },
              },
            ],
          }),
      } as Response);

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs: oneHourAgo,
        endTimeMs: now,
      });

      await waitFor(() => {
        // Check for locale-formatted number (1,234,567 in US locale)
        expect(screen.getByText('1,234,567')).toBeInTheDocument();
      });
    });
  });

  describe('API call parameters', () => {
    it('should call fetchOrFail with correct parameters', async () => {
      const startTimeMs = oneHourAgo;
      const endTimeMs = now;

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs,
        endTimeMs,
      });

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

    it('should calculate appropriate time interval for 1-hour range', async () => {
      const endTimeMs = Date.now();
      const startTimeMs = endTimeMs - 60 * 60 * 1000; // 1 hour

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs,
        endTimeMs,
      });

      await waitFor(() => {
        const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
        expect(callBody.time_interval_seconds).toBe(60); // MINUTE_IN_SECONDS
      });
    });

    it('should calculate appropriate time interval for 24-hour range', async () => {
      const endTimeMs = Date.now();
      const startTimeMs = endTimeMs - 12 * 60 * 60 * 1000; // 12 hours

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs,
        endTimeMs,
      });

      await waitFor(() => {
        const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
        expect(callBody.time_interval_seconds).toBe(3600); // HOUR_IN_SECONDS
      });
    });

    it('should calculate appropriate time interval for 7-day range', async () => {
      const endTimeMs = Date.now();
      const startTimeMs = endTimeMs - 7 * 24 * 60 * 60 * 1000; // 7 days

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs,
        endTimeMs,
      });

      await waitFor(() => {
        const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
        expect(callBody.time_interval_seconds).toBe(86400); // DAY_IN_SECONDS
      });
    });
  });

  describe('data transformation', () => {
    it('should handle data points with missing values gracefully', async () => {
      mockFetchOrFail.mockResolvedValue({
        json: () =>
          Promise.resolve({
            data_points: [
              {
                metric_name: TraceMetricKey.TRACE_COUNT,
                dimensions: { time_bucket: '2025-12-22T10:00:00Z' },
                values: {}, // Missing COUNT value
              },
              {
                metric_name: TraceMetricKey.TRACE_COUNT,
                dimensions: { time_bucket: '2025-12-22T11:00:00Z' },
                values: { [AggregationType.COUNT]: 50 },
              },
            ],
          }),
      } as Response);

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs: oneHourAgo,
        endTimeMs: now,
      });

      // Should still render and show total of 50 (0 + 50)
      await waitFor(() => {
        expect(screen.getByText('50')).toBeInTheDocument();
      });
    });

    it('should handle data points with missing time_bucket', async () => {
      mockFetchOrFail.mockResolvedValue({
        json: () =>
          Promise.resolve({
            data_points: [
              {
                metric_name: TraceMetricKey.TRACE_COUNT,
                dimensions: {}, // Missing time_bucket
                values: { [AggregationType.COUNT]: 25 },
              },
            ],
          }),
      } as Response);

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs: oneHourAgo,
        endTimeMs: now,
      });

      // Should still render with the count
      await waitFor(() => {
        expect(screen.getByText('25')).toBeInTheDocument();
      });
    });
  });
});
