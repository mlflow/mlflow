import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { AssessmentChartsSection } from './AssessmentChartsSection';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import {
  MetricViewType,
  AggregationType,
  AssessmentMetricKey,
  AssessmentFilterKey,
  AssessmentTypeValue,
  AssessmentDimensionKey,
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

// Helper to create an assessment data point with name and avg value
const createAssessmentDataPoint = (assessmentName: string, avgValue: number) => ({
  metric_name: AssessmentMetricKey.ASSESSMENT_VALUE,
  dimensions: { [AssessmentDimensionKey.ASSESSMENT_NAME]: assessmentName },
  values: { [AggregationType.AVG]: avgValue },
});

describe('AssessmentChartsSection', () => {
  const testExperimentId = 'test-experiment-123';
  const startTimeMs = new Date('2025-12-22T10:00:00Z').getTime();
  const endTimeMs = new Date('2025-12-22T12:00:00Z').getTime();
  const timeIntervalSeconds = 3600;

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
          <AssessmentChartsSection {...defaultProps} {...props} />
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

  describe('empty state', () => {
    it('should render empty state when no assessments are available', async () => {
      mockApiResponse([]);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('No assessments available')).toBeInTheDocument();
      });
    });
  });

  describe('with data', () => {
    const mockDataPoints = [
      createAssessmentDataPoint('Correctness', 0.85),
      createAssessmentDataPoint('Relevance', 0.72),
      createAssessmentDataPoint('Fluency', 0.9),
    ];

    it('should render section header with title', async () => {
      mockApiResponse(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Scorer Insights')).toBeInTheDocument();
      });
    });

    it('should render section description', async () => {
      mockApiResponse(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Quality metrics computed by scorers.')).toBeInTheDocument();
      });
    });

    it('should render a placeholder for each assessment', async () => {
      mockApiResponse(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        // Placeholders show the assessment names
        expect(screen.getByText('Correctness')).toBeInTheDocument();
        expect(screen.getByText('Fluency')).toBeInTheDocument();
        expect(screen.getByText('Relevance')).toBeInTheDocument();
      });
    });

    it('should display average values for each assessment', async () => {
      mockApiResponse(mockDataPoints);

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Avg: 0.85')).toBeInTheDocument();
        expect(screen.getByText('Avg: 0.72')).toBeInTheDocument();
        expect(screen.getByText('Avg: 0.90')).toBeInTheDocument();
      });
    });

    it('should sort assessments alphabetically', async () => {
      mockApiResponse([
        createAssessmentDataPoint('Zebra', 0.5),
        createAssessmentDataPoint('Alpha', 0.8),
        createAssessmentDataPoint('Middle', 0.6),
      ]);

      renderComponent();

      await waitFor(() => {
        // Check that all assessments are rendered
        expect(screen.getByText('Alpha')).toBeInTheDocument();
        expect(screen.getByText('Middle')).toBeInTheDocument();
        expect(screen.getByText('Zebra')).toBeInTheDocument();
      });
    });
  });

  describe('API call parameters', () => {
    it('should call API with correct parameters for fetching assessments', async () => {
      renderComponent();

      await waitFor(() => {
        const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
        expect(callBody).toMatchObject({
          experiment_ids: [testExperimentId],
          view_type: MetricViewType.ASSESSMENTS,
          metric_name: AssessmentMetricKey.ASSESSMENT_VALUE,
          aggregations: [{ aggregation_type: AggregationType.AVG }],
          dimensions: [AssessmentDimensionKey.ASSESSMENT_NAME],
          filters: [`assessment.${AssessmentFilterKey.TYPE} = "${AssessmentTypeValue.FEEDBACK}"`],
        });
      });
    });

    it('should include time range in API call', async () => {
      renderComponent();

      await waitFor(() => {
        const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
        expect(callBody.start_time_ms).toBe(startTimeMs);
        expect(callBody.end_time_ms).toBe(endTimeMs);
      });
    });
  });

  describe('data extraction', () => {
    it('should handle data points with missing assessment_name', async () => {
      mockApiResponse([
        createAssessmentDataPoint('ValidName', 0.8),
        {
          metric_name: AssessmentMetricKey.ASSESSMENT_VALUE,
          dimensions: {}, // Missing assessment_name
          values: { [AggregationType.AVG]: 0.5 },
        },
      ]);

      renderComponent();

      await waitFor(() => {
        // Should only render the valid assessment
        expect(screen.getByText('ValidName')).toBeInTheDocument();
        expect(screen.getByText('Avg: 0.80')).toBeInTheDocument();
      });
    });

    it('should handle data points with missing avg value', async () => {
      mockApiResponse([
        {
          metric_name: AssessmentMetricKey.ASSESSMENT_VALUE,
          dimensions: { [AssessmentDimensionKey.ASSESSMENT_NAME]: 'NoAvgValue' },
          values: {}, // Missing AVG value
        },
      ]);

      renderComponent();

      await waitFor(() => {
        // Should still render the assessment name
        expect(screen.getByText('NoAvgValue')).toBeInTheDocument();
        // Should show N/A for missing avg value
        expect(screen.getByText('Avg: N/A')).toBeInTheDocument();
      });
    });
  });
});
