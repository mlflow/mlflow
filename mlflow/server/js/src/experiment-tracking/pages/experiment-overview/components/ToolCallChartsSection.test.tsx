import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { ToolCallChartsSection } from './ToolCallChartsSection';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import {
  MetricViewType,
  AggregationType,
  SpanMetricKey,
  SpanFilterKey,
  SpanType,
  SpanStatus,
  SpanDimensionKey,
} from '@databricks/web-shared/model-trace-explorer';

// Mock FetchUtils
jest.mock('../../../../common/utils/FetchUtils', () => ({
  fetchOrFail: jest.fn(),
  getAjaxUrl: (url: string) => url,
}));

import { fetchOrFail } from '../../../../common/utils/FetchUtils';
const mockFetchOrFail = fetchOrFail as jest.MockedFunction<typeof fetchOrFail>;

// Mock LazyToolErrorRateChart to avoid nested async loading issues
jest.mock('./LazyToolErrorRateChart', () => ({
  LazyToolErrorRateChart: ({ toolName, overallErrorRate }: { toolName: string; overallErrorRate: number }) => (
    <div data-testid={`tool-chart-${toolName}`}>
      <span>{toolName}</span>
      <span>{overallErrorRate.toFixed(2)}%</span>
    </div>
  ),
}));

// Helper to create mock API response
const mockApiResponse = (dataPoints: any[]) => {
  mockFetchOrFail.mockResolvedValue({
    json: () => Promise.resolve({ data_points: dataPoints }),
  } as Response);
};

// Helper to create a data point with tool name and status
const createDataPoint = (toolName: string, status: string, count: number) => ({
  metric_name: SpanMetricKey.SPAN_COUNT,
  dimensions: {
    [SpanDimensionKey.SPAN_NAME]: toolName,
    [SpanDimensionKey.SPAN_STATUS]: status,
  },
  values: { [AggregationType.COUNT]: count },
});

describe('ToolCallChartsSection', () => {
  const testExperimentId = 'test-experiment-123';
  const startTimeMs = new Date('2025-12-22T10:00:00Z').getTime();
  const endTimeMs = new Date('2025-12-22T12:00:00Z').getTime();
  const timeIntervalSeconds = 3600;
  const timeBuckets = [new Date('2025-12-22T10:00:00Z').getTime(), new Date('2025-12-22T11:00:00Z').getTime()];

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
          <ToolCallChartsSection {...defaultProps} {...props} />
        </DesignSystemProvider>
      </QueryClientProvider>,
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('loading state', () => {
    it('should render loading state while data is being fetched', () => {
      mockFetchOrFail.mockReturnValue(new Promise(() => {})); // Never resolve

      renderComponent();

      // ChartLoadingState renders a spinner
      expect(screen.getByRole('img')).toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('should render error state when API call fails', async () => {
      mockFetchOrFail.mockRejectedValue(new Error('API Error'));

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Failed to load chart data')).toBeInTheDocument();
      });
    });
  });

  describe('empty state', () => {
    it('should render empty state when no tools are found', async () => {
      mockApiResponse([]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('No tool calls available')).toBeInTheDocument();
      });
    });
  });

  describe('with data', () => {
    it('should render a chart for each tool', async () => {
      mockApiResponse([
        createDataPoint('get_weather', SpanStatus.OK, 100),
        createDataPoint('search_docs', SpanStatus.OK, 50),
      ]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('tool-chart-get_weather')).toBeInTheDocument();
        expect(screen.getByTestId('tool-chart-search_docs')).toBeInTheDocument();
      });
    });

    it('should display tool names alphabetically sorted', async () => {
      mockApiResponse([
        createDataPoint('zebra_tool', SpanStatus.OK, 10),
        createDataPoint('alpha_tool', SpanStatus.OK, 20),
        createDataPoint('middle_tool', SpanStatus.OK, 30),
      ]);

      renderComponent();

      await waitFor(() => {
        const charts = screen.getAllByTestId(/^tool-chart-/);
        expect(charts[0]).toHaveAttribute('data-testid', 'tool-chart-alpha_tool');
        expect(charts[1]).toHaveAttribute('data-testid', 'tool-chart-middle_tool');
        expect(charts[2]).toHaveAttribute('data-testid', 'tool-chart-zebra_tool');
      });
    });

    it('should calculate correct error rate for each tool', async () => {
      // get_weather: 10 errors / 100 total = 10%
      // search_docs: 25 errors / 100 total = 25%
      mockApiResponse([
        createDataPoint('get_weather', SpanStatus.OK, 90),
        createDataPoint('get_weather', SpanStatus.ERROR, 10),
        createDataPoint('search_docs', SpanStatus.OK, 75),
        createDataPoint('search_docs', SpanStatus.ERROR, 25),
      ]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('10.00%')).toBeInTheDocument();
        expect(screen.getByText('25.00%')).toBeInTheDocument();
      });
    });

    it('should handle tools with 0% error rate', async () => {
      mockApiResponse([createDataPoint('perfect_tool', SpanStatus.OK, 100)]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('0.00%')).toBeInTheDocument();
      });
    });

    it('should handle tools with 100% error rate', async () => {
      mockApiResponse([createDataPoint('broken_tool', SpanStatus.ERROR, 50)]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('100.00%')).toBeInTheDocument();
      });
    });
  });

  describe('API call parameters', () => {
    it('should call API with correct view type and metric name', async () => {
      mockApiResponse([]);

      renderComponent();

      await waitFor(() => {
        expect(mockFetchOrFail).toHaveBeenCalled();
      });

      const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
      expect(callBody).toMatchObject({
        experiment_ids: [testExperimentId],
        view_type: MetricViewType.SPANS,
        metric_name: SpanMetricKey.SPAN_COUNT,
        aggregations: [{ aggregation_type: AggregationType.COUNT }],
      });
    });

    it('should filter by tool type', async () => {
      mockApiResponse([]);

      renderComponent();

      await waitFor(() => {
        expect(mockFetchOrFail).toHaveBeenCalled();
      });

      const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
      expect(callBody.filters).toContain(`span.${SpanFilterKey.TYPE} = "${SpanType.TOOL}"`);
    });

    it('should include span_name and span_status dimensions', async () => {
      mockApiResponse([]);

      renderComponent();

      await waitFor(() => {
        expect(mockFetchOrFail).toHaveBeenCalled();
      });

      const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
      expect(callBody.dimensions).toContain(SpanDimensionKey.SPAN_NAME);
      expect(callBody.dimensions).toContain(SpanDimensionKey.SPAN_STATUS);
    });

    it('should include time range in API call', async () => {
      mockApiResponse([]);

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
