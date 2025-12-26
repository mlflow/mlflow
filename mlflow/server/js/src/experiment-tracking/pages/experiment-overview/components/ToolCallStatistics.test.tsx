import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor, act } from '@testing-library/react';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { ToolCallStatistics } from './ToolCallStatistics';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import {
  MetricViewType,
  AggregationType,
  SpanMetricKey,
  SpanFilterKey,
  SpanType,
  SpanDimensionKey,
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

// Helper to create a count data point grouped by status
const createCountByStatusDataPoint = (status: string, count: number) => ({
  metric_name: SpanMetricKey.SPAN_COUNT,
  dimensions: { [SpanDimensionKey.SPAN_STATUS]: status },
  values: { [AggregationType.COUNT]: count },
});

// Helper to create a latency data point
const createLatencyDataPoint = (avgLatency: number) => ({
  metric_name: SpanMetricKey.LATENCY,
  dimensions: {},
  values: { [AggregationType.AVG]: avgLatency },
});

// Helper to mock both API calls (counts by status + latency)
const mockApiResponses = (countDataPoints: any[], latencyDataPoints: any[]) => {
  mockFetchOrFail
    .mockResolvedValueOnce({
      json: () => Promise.resolve({ data_points: countDataPoints }),
    } as Response)
    .mockResolvedValueOnce({
      json: () => Promise.resolve({ data_points: latencyDataPoints }),
    } as Response);
};

describe('ToolCallStatistics', () => {
  const testExperimentId = 'test-experiment-123';
  const startTimeMs = new Date('2025-12-22T10:00:00Z').getTime();
  const endTimeMs = new Date('2025-12-22T12:00:00Z').getTime();

  const defaultProps = {
    experimentId: testExperimentId,
    startTimeMs,
    endTimeMs,
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
          <ToolCallStatistics {...defaultProps} {...props} />
        </DesignSystemProvider>
      </QueryClientProvider>,
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('loading state', () => {
    it('should render loading spinners while data is being fetched', () => {
      mockFetchOrFail.mockReturnValue(new Promise(() => {})); // Never resolve

      renderComponent();

      // Should show spinners (4 stat cards, each with a spinner)
      const spinners = screen.getAllByRole('img');
      expect(spinners.length).toBeGreaterThanOrEqual(1);
    });
  });

  describe('with data', () => {
    it('should render all four stat cards', async () => {
      const countData = [createCountByStatusDataPoint('OK', 100), createCountByStatusDataPoint('ERROR', 5)];
      const latencyData = [createLatencyDataPoint(250)];
      mockApiResponses(countData, latencyData);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Total Tool Calls')).toBeInTheDocument();
        expect(screen.getByText('Success Rate')).toBeInTheDocument();
        expect(screen.getByText('Avg Latency')).toBeInTheDocument();
        expect(screen.getByText('Failed Calls')).toBeInTheDocument();
      });
    });

    it('should display correct total count', async () => {
      const countData = [createCountByStatusDataPoint('OK', 100), createCountByStatusDataPoint('ERROR', 5)];
      mockApiResponses(countData, [createLatencyDataPoint(0)]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('105')).toBeInTheDocument(); // 100 + 5 = 105
      });
    });

    it('should display correct success rate', async () => {
      const countData = [createCountByStatusDataPoint('OK', 95), createCountByStatusDataPoint('ERROR', 5)];
      mockApiResponses(countData, [createLatencyDataPoint(0)]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('95.00%')).toBeInTheDocument(); // 95/100 = 95%
      });
    });

    it('should display correct failed count', async () => {
      const countData = [createCountByStatusDataPoint('OK', 90), createCountByStatusDataPoint('ERROR', 10)];
      mockApiResponses(countData, [createLatencyDataPoint(0)]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('10')).toBeInTheDocument(); // 10 errors
      });
    });

    it('should display latency in milliseconds when < 1000ms', async () => {
      mockApiResponses([createCountByStatusDataPoint('OK', 100)], [createLatencyDataPoint(250.5)]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('250.50ms')).toBeInTheDocument();
      });
    });

    it('should display latency in seconds when >= 1000ms', async () => {
      mockApiResponses([createCountByStatusDataPoint('OK', 100)], [createLatencyDataPoint(1500)]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('1.50s')).toBeInTheDocument();
      });
    });

    it('should format large numbers with K suffix', async () => {
      const countData = [createCountByStatusDataPoint('OK', 5000)];
      mockApiResponses(countData, [createLatencyDataPoint(0)]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('5.0K')).toBeInTheDocument();
      });
    });

    it('should handle UNSET status in addition to OK and ERROR', async () => {
      const countData = [
        createCountByStatusDataPoint('OK', 80),
        createCountByStatusDataPoint('ERROR', 15),
        createCountByStatusDataPoint('UNSET', 5),
      ];
      mockApiResponses(countData, [createLatencyDataPoint(0)]);

      renderComponent();

      await waitFor(() => {
        // Total should include all statuses (80 + 15 + 5 = 100)
        expect(screen.getByText('100')).toBeInTheDocument();
        // Success rate based on OK only (80/100 = 80%)
        expect(screen.getByText('80.00%')).toBeInTheDocument();
        // Failed count based on ERROR only (15)
        expect(screen.getByText('15')).toBeInTheDocument();
      });
    });
  });

  describe('empty data', () => {
    it('should display zeros when no data is returned', async () => {
      mockApiResponses([], []);

      renderComponent();

      await waitFor(() => {
        // Should show 0 for counts
        const zeros = screen.getAllByText('0');
        expect(zeros.length).toBeGreaterThanOrEqual(2); // Total and Failed
        // Should show 0.00% for success rate
        expect(screen.getByText('0.00%')).toBeInTheDocument();
        // Should show 0.00ms for latency
        expect(screen.getByText('0.00ms')).toBeInTheDocument();
      });
    });
  });

  describe('API call parameters', () => {
    it('should call API with correct parameters for counts query', async () => {
      mockApiResponses([], []);

      renderComponent();

      await waitFor(() => {
        expect(mockFetchOrFail).toHaveBeenCalled();
      });

      // First call should be for counts grouped by status
      const firstCallBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
      expect(firstCallBody).toMatchObject({
        experiment_ids: [testExperimentId],
        view_type: MetricViewType.SPANS,
        metric_name: SpanMetricKey.SPAN_COUNT,
        aggregations: [{ aggregation_type: AggregationType.COUNT }],
        filters: [`span.${SpanFilterKey.TYPE} = "${SpanType.TOOL}"`],
        dimensions: [SpanDimensionKey.SPAN_STATUS],
      });
    });

    it('should call API with correct parameters for latency query', async () => {
      mockApiResponses([], []);

      renderComponent();

      await waitFor(() => {
        expect(mockFetchOrFail).toHaveBeenCalledTimes(2);
      });

      // Second call should be for latency
      const secondCallBody = JSON.parse((mockFetchOrFail.mock.calls[1]?.[1] as any)?.body || '{}');
      expect(secondCallBody).toMatchObject({
        experiment_ids: [testExperimentId],
        view_type: MetricViewType.SPANS,
        metric_name: SpanMetricKey.LATENCY,
        aggregations: [{ aggregation_type: AggregationType.AVG }],
        filters: [`span.${SpanFilterKey.TYPE} = "${SpanType.TOOL}"`],
      });
    });

    it('should include time range in API calls', async () => {
      mockApiResponses([], []);

      renderComponent();

      await waitFor(() => {
        expect(mockFetchOrFail).toHaveBeenCalled();
      });

      const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
      expect(callBody.start_time_ms).toBe(startTimeMs);
      expect(callBody.end_time_ms).toBe(endTimeMs);
    });
  });
});
