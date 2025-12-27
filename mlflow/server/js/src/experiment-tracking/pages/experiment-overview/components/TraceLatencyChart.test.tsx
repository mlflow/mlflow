import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { TraceLatencyChart } from './TraceLatencyChart';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import {
  MetricViewType,
  AggregationType,
  TraceMetricKey,
  P50,
  P90,
  P99,
  getPercentileKey,
} from '@databricks/web-shared/model-trace-explorer';

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

// Helper to chain mock API responses (for percentiles + AVG calls)
const mockApiResponses = (percentileDataPoints: any[], avgDataPoints: any[]) => {
  mockFetchOrFail
    .mockResolvedValueOnce({
      json: () => Promise.resolve({ data_points: percentileDataPoints }),
    } as Response)
    .mockResolvedValueOnce({
      json: () => Promise.resolve({ data_points: avgDataPoints }),
    } as Response);
};

// Helper to create a latency percentile data point
const createLatencyDataPoint = (timeBucket: string, p50: number, p90: number, p99: number) => ({
  metric_name: TraceMetricKey.LATENCY,
  dimensions: { time_bucket: timeBucket },
  values: {
    [getPercentileKey(P50)]: p50,
    [getPercentileKey(P90)]: p90,
    [getPercentileKey(P99)]: p99,
  },
});

// Helper to create an AVG latency data point
const createAvgLatencyDataPoint = (avg: number) => ({
  metric_name: TraceMetricKey.LATENCY,
  dimensions: {},
  values: { [AggregationType.AVG]: avg },
});

// Mock recharts components to avoid rendering issues in tests
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="responsive-container">{children}</div>
  ),
  LineChart: ({ children, data }: { children: React.ReactNode; data: any[] }) => (
    <div data-testid="line-chart" data-count={data?.length || 0}>
      {children}
    </div>
  ),
  Line: ({ name }: { name: string }) => <div data-testid={`line-${name}`} />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
  ReferenceLine: ({ label }: { label?: { value: string } }) => (
    <div data-testid="reference-line" data-label={label?.value} />
  ),
}));

describe('TraceLatencyChart', () => {
  const testExperimentId = 'test-experiment-123';
  const now = Date.now();
  const oneHourAgo = now - 60 * 60 * 1000;

  // Default props reused across tests
  const defaultProps = {
    experimentId: testExperimentId,
    startTimeMs: oneHourAgo,
    endTimeMs: now,
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
          <TraceLatencyChart {...defaultProps} {...props} />
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
    it('should render error message when time series API call fails', async () => {
      mockFetchOrFail.mockRejectedValue(new Error('API Error'));

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Failed to load chart data')).toBeInTheDocument();
      });
    });
  });

  describe('empty data state', () => {
    it('should render empty state message when no data points are returned', async () => {
      mockApiResponse([]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('No data available for the selected time range')).toBeInTheDocument();
      });
    });

    it('should render empty state when data_points is undefined', async () => {
      mockApiResponse(undefined);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('No data available for the selected time range')).toBeInTheDocument();
      });
    });
  });

  describe('with data', () => {
    const mockPercentileDataPoints = [
      createLatencyDataPoint('2025-12-22T10:00:00Z', 150, 350, 800),
      createLatencyDataPoint('2025-12-22T11:00:00Z', 180, 400, 950),
    ];

    const mockAvgDataPoints = [createAvgLatencyDataPoint(250)];

    it('should render chart with data points', async () => {
      mockApiResponses(mockPercentileDataPoints, mockAvgDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });

      // Verify the line chart has the correct number of data points
      expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', '2');
    });

    it('should display all three percentile lines', async () => {
      mockApiResponses(mockPercentileDataPoints, mockAvgDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('line-p50')).toBeInTheDocument();
        expect(screen.getByTestId('line-p90')).toBeInTheDocument();
        expect(screen.getByTestId('line-p99')).toBeInTheDocument();
      });
    });

    it('should display the average latency in header', async () => {
      mockApiResponses(mockPercentileDataPoints, mockAvgDataPoints);

      renderComponent();

      // Average is 250ms
      await waitFor(() => {
        expect(screen.getByText('250 ms')).toBeInTheDocument();
      });
    });

    it('should display the "Latency" title', async () => {
      mockApiResponses(mockPercentileDataPoints, mockAvgDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Latency')).toBeInTheDocument();
      });
    });

    it('should display "Over time" label', async () => {
      mockApiResponses(mockPercentileDataPoints, mockAvgDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Over time')).toBeInTheDocument();
      });
    });

    it('should format latency in seconds for values >= 1000ms', async () => {
      mockApiResponses(mockPercentileDataPoints, [createAvgLatencyDataPoint(1500)]);

      renderComponent();

      // 1500ms should be displayed as 1.50 sec
      await waitFor(() => {
        expect(screen.getByText('1.50 sec')).toBeInTheDocument();
      });
    });

    it('should render reference line with AVG label', async () => {
      mockApiResponses(mockPercentileDataPoints, mockAvgDataPoints);

      renderComponent();

      await waitFor(() => {
        const referenceLine = screen.getByTestId('reference-line');
        expect(referenceLine).toBeInTheDocument();
        expect(referenceLine).toHaveAttribute('data-label', 'AVG (250 ms)');
      });
    });
  });

  describe('API call parameters', () => {
    it('should call fetchOrFail for percentiles with correct parameters', async () => {
      renderComponent();

      await waitFor(() => {
        const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
        expect(callBody).toMatchObject({
          experiment_ids: [testExperimentId],
          view_type: MetricViewType.TRACES,
          metric_name: TraceMetricKey.LATENCY,
          aggregations: [
            { aggregation_type: AggregationType.PERCENTILE, percentile_value: P50 },
            { aggregation_type: AggregationType.PERCENTILE, percentile_value: P90 },
            { aggregation_type: AggregationType.PERCENTILE, percentile_value: P99 },
          ],
        });
      });
    });

    it('should call fetchOrFail for AVG with correct parameters', async () => {
      renderComponent();

      await waitFor(() => {
        // Second call is for AVG
        const callBody = JSON.parse((mockFetchOrFail.mock.calls[1]?.[1] as any)?.body || '{}');
        expect(callBody).toMatchObject({
          experiment_ids: [testExperimentId],
          view_type: MetricViewType.TRACES,
          metric_name: TraceMetricKey.LATENCY,
          aggregations: [{ aggregation_type: AggregationType.AVG }],
        });
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
    it('should handle data points with missing percentile values gracefully', async () => {
      mockApiResponses(
        [
          {
            metric_name: TraceMetricKey.LATENCY,
            dimensions: { time_bucket: '2025-12-22T10:00:00Z' },
            values: {}, // Missing percentile values
          },
        ],
        [createAvgLatencyDataPoint(100)],
      );

      renderComponent();

      // Should still render without crashing
      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });
    });

    it('should handle missing AVG data gracefully', async () => {
      mockApiResponses([createLatencyDataPoint('2025-12-22T10:00:00Z', 150, 350, 800)], []);

      renderComponent();

      // Should still render the chart without avg
      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });

      // Should NOT display avg value or reference line when not available
      expect(screen.queryByTestId('reference-line')).not.toBeInTheDocument();
    });

    it('should handle data points with missing time_bucket', async () => {
      mockApiResponses(
        [
          {
            metric_name: TraceMetricKey.LATENCY,
            dimensions: {}, // Missing time_bucket
            values: {
              [getPercentileKey(P50)]: 100,
              [getPercentileKey(P90)]: 200,
              [getPercentileKey(P99)]: 300,
            },
          },
        ],
        [createAvgLatencyDataPoint(150)],
      );

      renderComponent();

      // Should still render the chart
      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });
    });
  });
});
