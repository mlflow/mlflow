import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { TraceLatencyChart } from './TraceLatencyChart';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MlflowService } from '../../../sdk/MlflowService';
import {
  MetricViewType,
  AggregationType,
  TraceMetricKey,
  P50,
  P90,
  P99,
  getPercentileKey,
} from '@databricks/web-shared/model-trace-explorer';

// Mock MlflowService
jest.mock('../../../sdk/MlflowService', () => ({
  MlflowService: {
    queryTraceMetrics: jest.fn(),
  },
}));

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

const mockQueryTraceMetrics = MlflowService.queryTraceMetrics as jest.MockedFunction<
  typeof MlflowService.queryTraceMetrics
>;

describe('TraceLatencyChart', () => {
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
          <TraceLatencyChart {...props} />
        </DesignSystemProvider>
      </QueryClientProvider>,
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('loading state', () => {
    it('should render loading spinner while data is being fetched', async () => {
      // Create a promise that never resolves to keep the component in loading state
      mockQueryTraceMetrics.mockReturnValue(new Promise(() => {}));

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
    it('should render error message when time series API call fails', async () => {
      mockQueryTraceMetrics.mockRejectedValue(new Error('API Error'));

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
      mockQueryTraceMetrics.mockResolvedValue({
        data_points: [],
      });

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
      mockQueryTraceMetrics.mockResolvedValue({ data_points: undefined } as any);

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
    const mockPercentileDataPoints = [
      {
        metric_name: TraceMetricKey.LATENCY,
        dimensions: { time_bucket: '2025-12-22T10:00:00Z' },
        values: {
          [getPercentileKey(P50)]: 150,
          [getPercentileKey(P90)]: 350,
          [getPercentileKey(P99)]: 800,
        },
      },
      {
        metric_name: TraceMetricKey.LATENCY,
        dimensions: { time_bucket: '2025-12-22T11:00:00Z' },
        values: {
          [getPercentileKey(P50)]: 180,
          [getPercentileKey(P90)]: 400,
          [getPercentileKey(P99)]: 950,
        },
      },
    ];

    const mockAvgDataPoints = [
      {
        metric_name: TraceMetricKey.LATENCY,
        dimensions: {},
        values: { [AggregationType.AVG]: 250 },
      },
    ];

    it('should render chart with data points', async () => {
      mockQueryTraceMetrics
        .mockResolvedValueOnce({ data_points: mockPercentileDataPoints })
        .mockResolvedValueOnce({ data_points: mockAvgDataPoints });

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs: oneHourAgo,
        endTimeMs: now,
      });

      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });

      // Verify the line chart has the correct number of data points
      expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', '2');
    });

    it('should display all three percentile lines', async () => {
      mockQueryTraceMetrics
        .mockResolvedValueOnce({ data_points: mockPercentileDataPoints })
        .mockResolvedValueOnce({ data_points: mockAvgDataPoints });

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs: oneHourAgo,
        endTimeMs: now,
      });

      await waitFor(() => {
        expect(screen.getByTestId('line-p50')).toBeInTheDocument();
        expect(screen.getByTestId('line-p90')).toBeInTheDocument();
        expect(screen.getByTestId('line-p99')).toBeInTheDocument();
      });
    });

    it('should display the average latency in header', async () => {
      mockQueryTraceMetrics
        .mockResolvedValueOnce({ data_points: mockPercentileDataPoints })
        .mockResolvedValueOnce({ data_points: mockAvgDataPoints });

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs: oneHourAgo,
        endTimeMs: now,
      });

      // Average is 250ms
      await waitFor(() => {
        expect(screen.getByText('250 ms')).toBeInTheDocument();
      });
    });

    it('should display the "Latency" title', async () => {
      mockQueryTraceMetrics
        .mockResolvedValueOnce({ data_points: mockPercentileDataPoints })
        .mockResolvedValueOnce({ data_points: mockAvgDataPoints });

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs: oneHourAgo,
        endTimeMs: now,
      });

      await waitFor(() => {
        expect(screen.getByText('Latency')).toBeInTheDocument();
      });
    });

    it('should display "Over time" label', async () => {
      mockQueryTraceMetrics
        .mockResolvedValueOnce({ data_points: mockPercentileDataPoints })
        .mockResolvedValueOnce({ data_points: mockAvgDataPoints });

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs: oneHourAgo,
        endTimeMs: now,
      });

      await waitFor(() => {
        expect(screen.getByText('Over time')).toBeInTheDocument();
      });
    });

    it('should format latency in seconds for values >= 1000ms', async () => {
      mockQueryTraceMetrics.mockResolvedValueOnce({ data_points: mockPercentileDataPoints }).mockResolvedValueOnce({
        data_points: [
          {
            metric_name: TraceMetricKey.LATENCY,
            dimensions: {},
            values: { [AggregationType.AVG]: 1500 },
          },
        ],
      });

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs: oneHourAgo,
        endTimeMs: now,
      });

      // 1500ms should be displayed as 1.50 sec
      await waitFor(() => {
        expect(screen.getByText('1.50 sec')).toBeInTheDocument();
      });
    });

    it('should render reference line with AVG label', async () => {
      mockQueryTraceMetrics
        .mockResolvedValueOnce({ data_points: mockPercentileDataPoints })
        .mockResolvedValueOnce({ data_points: mockAvgDataPoints });

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs: oneHourAgo,
        endTimeMs: now,
      });

      await waitFor(() => {
        const referenceLine = screen.getByTestId('reference-line');
        expect(referenceLine).toBeInTheDocument();
        expect(referenceLine).toHaveAttribute('data-label', 'AVG (250 ms)');
      });
    });
  });

  describe('API call parameters', () => {
    it('should call queryTraceMetrics for percentiles with correct parameters', async () => {
      mockQueryTraceMetrics.mockResolvedValue({ data_points: [] });

      const startTimeMs = oneHourAgo;
      const endTimeMs = now;

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs,
        endTimeMs,
      });

      await waitFor(() => {
        expect(mockQueryTraceMetrics).toHaveBeenCalledWith(
          expect.objectContaining({
            experiment_ids: [testExperimentId],
            view_type: MetricViewType.TRACES,
            metric_name: TraceMetricKey.LATENCY,
            aggregations: [
              { aggregation_type: AggregationType.PERCENTILE, percentile_value: P50 },
              { aggregation_type: AggregationType.PERCENTILE, percentile_value: P90 },
              { aggregation_type: AggregationType.PERCENTILE, percentile_value: P99 },
            ],
            start_time_ms: startTimeMs,
            end_time_ms: endTimeMs,
          }),
        );
      });
    });

    it('should call queryTraceMetrics for AVG with correct parameters', async () => {
      mockQueryTraceMetrics.mockResolvedValue({ data_points: [] });

      const startTimeMs = oneHourAgo;
      const endTimeMs = now;

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs,
        endTimeMs,
      });

      await waitFor(() => {
        expect(mockQueryTraceMetrics).toHaveBeenCalledWith(
          expect.objectContaining({
            experiment_ids: [testExperimentId],
            view_type: MetricViewType.TRACES,
            metric_name: TraceMetricKey.LATENCY,
            aggregations: [{ aggregation_type: AggregationType.AVG }],
            start_time_ms: startTimeMs,
            end_time_ms: endTimeMs,
          }),
        );
      });
    });

    it('should calculate appropriate time interval for 1-hour range', async () => {
      mockQueryTraceMetrics.mockResolvedValue({ data_points: [] });

      const endTimeMs = Date.now();
      const startTimeMs = endTimeMs - 60 * 60 * 1000; // 1 hour

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs,
        endTimeMs,
      });

      await waitFor(() => {
        expect(mockQueryTraceMetrics).toHaveBeenCalledWith(
          expect.objectContaining({
            time_interval_seconds: 60, // MINUTE_IN_SECONDS
          }),
        );
      });
    });

    it('should calculate appropriate time interval for 12-hour range', async () => {
      mockQueryTraceMetrics.mockResolvedValue({ data_points: [] });

      const endTimeMs = Date.now();
      const startTimeMs = endTimeMs - 12 * 60 * 60 * 1000; // 12 hours

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs,
        endTimeMs,
      });

      await waitFor(() => {
        expect(mockQueryTraceMetrics).toHaveBeenCalledWith(
          expect.objectContaining({
            time_interval_seconds: 3600, // HOUR_IN_SECONDS
          }),
        );
      });
    });

    it('should calculate appropriate time interval for 7-day range', async () => {
      mockQueryTraceMetrics.mockResolvedValue({ data_points: [] });

      const endTimeMs = Date.now();
      const startTimeMs = endTimeMs - 7 * 24 * 60 * 60 * 1000; // 7 days

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs,
        endTimeMs,
      });

      await waitFor(() => {
        expect(mockQueryTraceMetrics).toHaveBeenCalledWith(
          expect.objectContaining({
            time_interval_seconds: 86400, // DAY_IN_SECONDS
          }),
        );
      });
    });
  });

  describe('data transformation', () => {
    it('should handle data points with missing percentile values gracefully', async () => {
      mockQueryTraceMetrics
        .mockResolvedValueOnce({
          data_points: [
            {
              metric_name: TraceMetricKey.LATENCY,
              dimensions: { time_bucket: '2025-12-22T10:00:00Z' },
              values: {}, // Missing percentile values
            },
          ],
        })
        .mockResolvedValueOnce({
          data_points: [
            {
              metric_name: TraceMetricKey.LATENCY,
              dimensions: {},
              values: { [AggregationType.AVG]: 100 },
            },
          ],
        });

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs: oneHourAgo,
        endTimeMs: now,
      });

      // Should still render without crashing
      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });
    });

    it('should handle missing AVG data gracefully', async () => {
      mockQueryTraceMetrics
        .mockResolvedValueOnce({
          data_points: [
            {
              metric_name: TraceMetricKey.LATENCY,
              dimensions: { time_bucket: '2025-12-22T10:00:00Z' },
              values: {
                [getPercentileKey(P50)]: 150,
                [getPercentileKey(P90)]: 350,
                [getPercentileKey(P99)]: 800,
              },
            },
          ],
        })
        .mockResolvedValueOnce({ data_points: [] });

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs: oneHourAgo,
        endTimeMs: now,
      });

      // Should still render the chart without avg
      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });

      // Should NOT display avg value or reference line when not available
      expect(screen.queryByTestId('reference-line')).not.toBeInTheDocument();
    });

    it('should handle data points with missing time_bucket', async () => {
      mockQueryTraceMetrics
        .mockResolvedValueOnce({
          data_points: [
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
        })
        .mockResolvedValueOnce({
          data_points: [
            {
              metric_name: TraceMetricKey.LATENCY,
              dimensions: {},
              values: { [AggregationType.AVG]: 150 },
            },
          ],
        });

      renderComponent({
        experimentId: testExperimentId,
        startTimeMs: oneHourAgo,
        endTimeMs: now,
      });

      // Should still render the chart
      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });
    });
  });
});
