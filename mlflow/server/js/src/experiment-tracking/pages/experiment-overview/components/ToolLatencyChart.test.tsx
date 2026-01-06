import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { ToolLatencyChart } from './ToolLatencyChart';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { AggregationType, SpanMetricKey, SpanDimensionKey } from '@databricks/web-shared/model-trace-explorer';

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

describe('ToolLatencyChart', () => {
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

  const renderComponent = (props: Partial<typeof defaultProps> = {}) => {
    const queryClient = createQueryClient();
    return renderWithIntl(
      <QueryClientProvider client={queryClient}>
        <DesignSystemProvider>
          <ToolLatencyChart {...defaultProps} {...props} />
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
      // Suppress expected console.error from react-query
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

      mockFetchOrFail.mockRejectedValue(new Error('API Error'));

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Failed to load chart data')).toBeInTheDocument();
      });

      consoleSpy.mockRestore();
    });
  });

  describe('empty data state', () => {
    it('should render empty state when no data points are returned', async () => {
      mockApiResponse([]);

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
    const mockDataPoints = [
      createToolLatencyDataPoint('2025-12-22T10:00:00Z', 'delivery_estimate', 150),
      createToolLatencyDataPoint('2025-12-22T11:00:00Z', 'delivery_estimate', 180),
      createToolLatencyDataPoint('2025-12-22T12:00:00Z', 'delivery_estimate', 200),
      createToolLatencyDataPoint('2025-12-22T10:00:00Z', 'get_order_status', 120),
      createToolLatencyDataPoint('2025-12-22T11:00:00Z', 'get_order_status', 140),
    ];

    it('should render chart when data is available', async () => {
      mockApiResponse(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });

      expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', '3');
    });

    it('should display the chart title', async () => {
      mockApiResponse(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Latency Comparison')).toBeInTheDocument();
      });
    });

    it('should display "Over time" label', async () => {
      mockApiResponse(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Over time')).toBeInTheDocument();
      });
    });
  });

  describe('with single tool', () => {
    it('should render chart with single tool data', async () => {
      mockApiResponse([
        createToolLatencyDataPoint('2025-12-22T10:00:00Z', 'single_tool', 100),
        createToolLatencyDataPoint('2025-12-22T11:00:00Z', 'single_tool', 150),
      ]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });

      expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', '3');
    });
  });
});
