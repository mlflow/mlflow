import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { TraceAssessmentChart } from './TraceAssessmentChart';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import {
  MetricViewType,
  AggregationType,
  AssessmentMetricKey,
  AssessmentFilterKey,
  AssessmentDimensionKey,
} from '@databricks/web-shared/model-trace-explorer';

// Mock FetchUtils
jest.mock('../../../../common/utils/FetchUtils', () => ({
  fetchOrFail: jest.fn(),
  getAjaxUrl: (url: string) => url,
}));

import { fetchOrFail } from '../../../../common/utils/FetchUtils';
const mockFetchOrFail = fetchOrFail as jest.MockedFunction<typeof fetchOrFail>;

// Helper to create mock API response for a single call
const mockApiResponse = (dataPoints: any[] | undefined) => {
  mockFetchOrFail.mockResolvedValue({
    json: () => Promise.resolve({ data_points: dataPoints }),
  } as Response);
};

// Helper to mock different responses for time series and distribution queries
const mockApiResponses = (timeSeriesData: any[], distributionData: any[]) => {
  mockFetchOrFail
    .mockResolvedValueOnce({
      json: () => Promise.resolve({ data_points: timeSeriesData }),
    } as Response)
    .mockResolvedValueOnce({
      json: () => Promise.resolve({ data_points: distributionData }),
    } as Response);
};

// Helper to create an assessment value data point (for time series)
const createAssessmentDataPoint = (timeBucket: string, avgValue: number) => ({
  metric_name: AssessmentMetricKey.ASSESSMENT_VALUE,
  dimensions: { time_bucket: timeBucket },
  values: { [AggregationType.AVG]: avgValue },
});

// Helper to create a distribution data point (for bar chart)
const createDistributionDataPoint = (assessmentValue: string, count: number) => ({
  metric_name: AssessmentMetricKey.ASSESSMENT_COUNT,
  dimensions: { [AssessmentDimensionKey.ASSESSMENT_VALUE]: assessmentValue },
  values: { [AggregationType.COUNT]: count },
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
  BarChart: ({ children, data }: { children: React.ReactNode; data: any[] }) => (
    <div data-testid="bar-chart" data-count={data?.length || 0} data-labels={data?.map((d) => d.name).join(',')}>
      {children}
    </div>
  ),
  Line: ({ dataKey }: { dataKey: string }) => <div data-testid={`line-${dataKey}`} />,
  Bar: ({ dataKey }: { dataKey: string }) => <div data-testid={`bar-${dataKey}`} />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
  ReferenceLine: ({ label }: { label?: { value: string } }) => (
    <div data-testid="reference-line" data-label={label?.value} />
  ),
}));

describe('TraceAssessmentChart', () => {
  const testExperimentId = 'test-experiment-123';
  const testAssessmentName = 'Correctness';
  // Use fixed timestamps for predictable bucket generation
  const startTimeMs = new Date('2025-12-22T10:00:00Z').getTime();
  const endTimeMs = new Date('2025-12-22T12:00:00Z').getTime();
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
    assessmentName: testAssessmentName,
  };

  const createQueryClient = () =>
    new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

  const renderComponent = (props: Partial<typeof defaultProps & { lineColor?: string; avgValue?: number }> = {}) => {
    const queryClient = createQueryClient();
    return renderWithIntl(
      <QueryClientProvider client={queryClient}>
        <DesignSystemProvider>
          <TraceAssessmentChart {...defaultProps} {...props} />
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
    it('should render empty state when no data points are returned', async () => {
      mockApiResponse([]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('No data available for the selected time range')).toBeInTheDocument();
      });
    });
  });

  describe('with data', () => {
    const mockDataPoints = [
      createAssessmentDataPoint('2025-12-22T10:00:00Z', 0.75),
      createAssessmentDataPoint('2025-12-22T11:00:00Z', 0.82),
    ];

    it('should render chart with all time buckets', async () => {
      mockApiResponse(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });

      // Verify the line chart has all 3 time buckets
      expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', '3');
    });

    it('should display the assessment name as title', async () => {
      mockApiResponse(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText(testAssessmentName)).toBeInTheDocument();
      });
    });

    it('should display chart section labels', async () => {
      mockApiResponse(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Total aggregate scores')).toBeInTheDocument();
        expect(screen.getByText('Moving average over time')).toBeInTheDocument();
      });
    });

    it('should display average value when provided via prop', async () => {
      mockApiResponse(mockDataPoints);

      renderComponent({ avgValue: 0.78 });

      await waitFor(() => {
        expect(screen.getByText('0.78')).toBeInTheDocument();
        expect(screen.getByText('avg score')).toBeInTheDocument();
      });
    });

    it('should render reference line when avgValue is provided', async () => {
      mockApiResponse(mockDataPoints);

      renderComponent({ avgValue: 0.78 });

      await waitFor(() => {
        const referenceLine = screen.getByTestId('reference-line');
        expect(referenceLine).toBeInTheDocument();
        expect(referenceLine).toHaveAttribute('data-label', 'AVG (0.78)');
      });
    });

    it('should NOT render reference line when avgValue is not provided', async () => {
      mockApiResponse(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });

      expect(screen.queryByTestId('reference-line')).not.toBeInTheDocument();
    });

    it('should fill missing time buckets with zeros', async () => {
      // Only provide data for one time bucket
      mockApiResponse([createAssessmentDataPoint('2025-12-22T10:00:00Z', 0.8)]);

      renderComponent();

      // Chart should still show all 3 time buckets
      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', '3');
      });
    });
  });

  describe('API call parameters', () => {
    it('should call fetchOrFail with correct parameters', async () => {
      renderComponent();

      await waitFor(() => {
        const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
        expect(callBody).toMatchObject({
          experiment_ids: [testExperimentId],
          view_type: MetricViewType.ASSESSMENTS,
          metric_name: AssessmentMetricKey.ASSESSMENT_VALUE,
          aggregations: [{ aggregation_type: AggregationType.AVG }],
          filters: [`assessment.${AssessmentFilterKey.NAME} = "${testAssessmentName}"`],
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
  });

  describe('custom props', () => {
    it('should accept custom lineColor', async () => {
      mockApiResponse([createAssessmentDataPoint('2025-12-22T10:00:00Z', 0.8)]);

      renderComponent({ lineColor: '#FF0000' });

      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });
    });

    it('should render with different assessment names', async () => {
      mockApiResponse([createAssessmentDataPoint('2025-12-22T10:00:00Z', 0.9)]);

      renderComponent({ assessmentName: 'Relevance' });

      await waitFor(() => {
        expect(screen.getByText('Relevance')).toBeInTheDocument();
      });
    });
  });

  describe('distribution chart - bucketing behavior', () => {
    it('should NOT bucket integer values with 5 or fewer unique values', async () => {
      const timeSeriesData = [createAssessmentDataPoint('2025-12-22T10:00:00Z', 3.0)];
      const distributionData = [
        createDistributionDataPoint('1', 5),
        createDistributionDataPoint('2', 10),
        createDistributionDataPoint('3', 15),
        createDistributionDataPoint('4', 8),
        createDistributionDataPoint('5', 3),
      ];

      mockApiResponses(timeSeriesData, distributionData);
      renderComponent();

      await waitFor(() => {
        const barChart = screen.getByTestId('bar-chart');
        expect(barChart).toBeInTheDocument();
        // Should show individual values, not bucketed
        expect(barChart).toHaveAttribute('data-count', '5');
        expect(barChart).toHaveAttribute('data-labels', '1,2,3,4,5');
      });
    });

    it('should bucket integer values with more than 5 unique values', async () => {
      const timeSeriesData = [createAssessmentDataPoint('2025-12-22T10:00:00Z', 5.0)];
      const distributionData = [
        createDistributionDataPoint('1', 5),
        createDistributionDataPoint('2', 10),
        createDistributionDataPoint('3', 15),
        createDistributionDataPoint('4', 8),
        createDistributionDataPoint('5', 3),
        createDistributionDataPoint('6', 7),
        createDistributionDataPoint('7', 2),
      ];

      mockApiResponses(timeSeriesData, distributionData);
      renderComponent();

      await waitFor(() => {
        const barChart = screen.getByTestId('bar-chart');
        expect(barChart).toBeInTheDocument();
        // Should be bucketed into 5 ranges
        expect(barChart).toHaveAttribute('data-count', '5');
      });
    });

    it('should bucket float values regardless of count', async () => {
      const timeSeriesData = [createAssessmentDataPoint('2025-12-22T10:00:00Z', 0.75)];
      const distributionData = [
        createDistributionDataPoint('0.1', 5),
        createDistributionDataPoint('0.5', 10),
        createDistributionDataPoint('0.9', 8),
      ];

      mockApiResponses(timeSeriesData, distributionData);
      renderComponent();

      await waitFor(() => {
        const barChart = screen.getByTestId('bar-chart');
        expect(barChart).toBeInTheDocument();
        // Should be bucketed into 5 ranges even with few unique values
        expect(barChart).toHaveAttribute('data-count', '5');
      });
    });

    it('should NOT bucket boolean values', async () => {
      const timeSeriesData = [createAssessmentDataPoint('2025-12-22T10:00:00Z', 0.8)];
      const distributionData = [createDistributionDataPoint('true', 15), createDistributionDataPoint('false', 5)];

      mockApiResponses(timeSeriesData, distributionData);
      renderComponent();

      await waitFor(() => {
        const barChart = screen.getByTestId('bar-chart');
        expect(barChart).toBeInTheDocument();
        // Should show individual values
        expect(barChart).toHaveAttribute('data-count', '2');
        expect(barChart).toHaveAttribute('data-labels', 'false,true');
      });
    });

    it('should NOT bucket string values', async () => {
      const timeSeriesData = [createAssessmentDataPoint('2025-12-22T10:00:00Z', 0.8)];
      const distributionData = [
        createDistributionDataPoint('pass', 15),
        createDistributionDataPoint('fail', 5),
        createDistributionDataPoint('error', 2),
      ];

      mockApiResponses(timeSeriesData, distributionData);
      renderComponent();

      await waitFor(() => {
        const barChart = screen.getByTestId('bar-chart');
        expect(barChart).toBeInTheDocument();
        // Should show individual values sorted alphabetically
        expect(barChart).toHaveAttribute('data-count', '3');
        expect(barChart).toHaveAttribute('data-labels', 'error,fail,pass');
      });
    });

    it('should render both bar chart and line chart', async () => {
      const timeSeriesData = [createAssessmentDataPoint('2025-12-22T10:00:00Z', 0.8)];
      const distributionData = [createDistributionDataPoint('0.8', 10)];

      mockApiResponses(timeSeriesData, distributionData);
      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });
    });

    it('should display "Total aggregate scores" label for bar chart', async () => {
      const timeSeriesData = [createAssessmentDataPoint('2025-12-22T10:00:00Z', 0.8)];
      const distributionData = [createDistributionDataPoint('0.8', 10)];

      mockApiResponses(timeSeriesData, distributionData);
      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Total aggregate scores')).toBeInTheDocument();
      });
    });

    it('should display "Moving average over time" label for line chart', async () => {
      const timeSeriesData = [createAssessmentDataPoint('2025-12-22T10:00:00Z', 0.8)];
      const distributionData = [createDistributionDataPoint('0.8', 10)];

      mockApiResponses(timeSeriesData, distributionData);
      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Moving average over time')).toBeInTheDocument();
      });
    });
  });
});
