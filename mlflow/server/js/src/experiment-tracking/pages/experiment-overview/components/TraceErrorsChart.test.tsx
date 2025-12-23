import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { TraceErrorsChart } from './TraceErrorsChart';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { AggregationType, TraceMetricKey } from '@databricks/web-shared/model-trace-explorer';

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

// Helper to chain mock API responses (for error count + total count calls)
const mockApiResponses = (errorDataPoints: any[], totalDataPoints: any[]) => {
  mockFetchOrFail
    .mockResolvedValueOnce({
      json: () => Promise.resolve({ data_points: errorDataPoints }),
    } as Response)
    .mockResolvedValueOnce({
      json: () => Promise.resolve({ data_points: totalDataPoints }),
    } as Response);
};

// Helper to create an error count data point
const createErrorCountDataPoint = (timeBucket: string, count: number) => ({
  metric_name: TraceMetricKey.TRACE_COUNT,
  dimensions: { time_bucket: timeBucket },
  values: { [AggregationType.COUNT]: count },
});

// Helper to create a total count data point
const createTotalCountDataPoint = (timeBucket: string, count: number) => ({
  metric_name: TraceMetricKey.TRACE_COUNT,
  dimensions: { time_bucket: timeBucket },
  values: { [AggregationType.COUNT]: count },
});

// Mock recharts components to avoid rendering issues in tests
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="responsive-container">{children}</div>
  ),
  ComposedChart: ({ children, data }: { children: React.ReactNode; data: any[] }) => (
    <div data-testid="composed-chart" data-count={data?.length || 0}>
      {children}
    </div>
  ),
  Bar: ({ name }: { name: string }) => <div data-testid={`bar-${name}`} />,
  Line: ({ name }: { name: string }) => <div data-testid={`line-${name}`} />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
  ReferenceLine: ({ label }: { label?: { value: string } }) => (
    <div data-testid="reference-line" data-label={label?.value} />
  ),
}));

describe('TraceErrorsChart', () => {
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
          <TraceErrorsChart {...defaultProps} {...props} />
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
      mockFetchOrFail.mockReturnValue(new Promise(() => {}));

      renderComponent();

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
    const mockErrorDataPoints = [
      createErrorCountDataPoint('2025-12-22T10:00:00Z', 5),
      createErrorCountDataPoint('2025-12-22T11:00:00Z', 10),
    ];

    const mockTotalDataPoints = [
      createTotalCountDataPoint('2025-12-22T10:00:00Z', 100),
      createTotalCountDataPoint('2025-12-22T11:00:00Z', 200),
    ];

    it('should render chart with data points', async () => {
      mockApiResponses(mockErrorDataPoints, mockTotalDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('composed-chart')).toBeInTheDocument();
      });

      expect(screen.getByTestId('composed-chart')).toHaveAttribute('data-count', '2');
    });

    it('should display the "Errors" title', async () => {
      mockApiResponses(mockErrorDataPoints, mockTotalDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Errors')).toBeInTheDocument();
      });
    });

    it('should display the total error count', async () => {
      mockApiResponses(mockErrorDataPoints, mockTotalDataPoints);

      renderComponent();

      // Total errors should be 5 + 10 = 15
      await waitFor(() => {
        expect(screen.getByText(/15/)).toBeInTheDocument();
      });
    });

    it('should display the overall error rate', async () => {
      mockApiResponses(mockErrorDataPoints, mockTotalDataPoints);

      renderComponent();

      // Error rate: (5 + 10) / (100 + 200) = 15/300 = 5%
      await waitFor(() => {
        expect(screen.getByText(/Overall error rate: 5\.0%/)).toBeInTheDocument();
      });
    });

    it('should display "Over time" label', async () => {
      mockApiResponses(mockErrorDataPoints, mockTotalDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Over time')).toBeInTheDocument();
      });
    });

    it('should render both bar and line series', async () => {
      mockApiResponses(mockErrorDataPoints, mockTotalDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('bar-Error Count')).toBeInTheDocument();
        expect(screen.getByTestId('line-Error Rate')).toBeInTheDocument();
      });
    });

    it('should render reference line with AVG label', async () => {
      mockApiResponses(mockErrorDataPoints, mockTotalDataPoints);

      renderComponent();

      await waitFor(() => {
        const referenceLine = screen.getByTestId('reference-line');
        expect(referenceLine).toBeInTheDocument();
        // AVG error rate: (5% + 5%) / 2 = 5%
        expect(referenceLine).toHaveAttribute('data-label', expect.stringContaining('AVG'));
      });
    });
  });

  describe('API call parameters', () => {
    it('should call fetchOrFail with error filter for first call', async () => {
      renderComponent();

      await waitFor(() => {
        const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
        expect(callBody.filters).toContain('trace.status = "ERROR"');
      });
    });

    it('should call fetchOrFail without filter for total count call', async () => {
      renderComponent();

      await waitFor(() => {
        // Second call is for total count (no error filter)
        const callBody = JSON.parse((mockFetchOrFail.mock.calls[1]?.[1] as any)?.body || '{}');
        expect(callBody.filters).toBeUndefined();
      });
    });

    it('should use provided time interval', async () => {
      renderComponent({ timeIntervalSeconds: 60 });

      await waitFor(() => {
        const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
        expect(callBody.time_interval_seconds).toBe(60);
      });
    });
  });

  describe('data transformation', () => {
    it('should handle data points with missing values gracefully', async () => {
      mockApiResponses(
        [
          {
            metric_name: TraceMetricKey.TRACE_COUNT,
            dimensions: { time_bucket: '2025-12-22T10:00:00Z' },
            values: {}, // Missing COUNT value
          },
          createErrorCountDataPoint('2025-12-22T11:00:00Z', 10),
        ],
        [
          createTotalCountDataPoint('2025-12-22T10:00:00Z', 100),
          createTotalCountDataPoint('2025-12-22T11:00:00Z', 200),
        ],
      );

      renderComponent();

      // Should still render and show total of 10 (0 + 10)
      await waitFor(() => {
        expect(screen.getByText(/10/)).toBeInTheDocument();
      });
    });

    it('should handle zero total count gracefully (no division by zero)', async () => {
      mockApiResponses(
        [createErrorCountDataPoint('2025-12-22T10:00:00Z', 0)],
        [createTotalCountDataPoint('2025-12-22T10:00:00Z', 0)],
      );

      renderComponent();

      // Should render without crashing, error rate should be 0%
      await waitFor(() => {
        expect(screen.getByText(/Overall error rate: 0\.0%/)).toBeInTheDocument();
      });
    });
  });
});
