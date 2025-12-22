import { jest, describe, it, expect, beforeEach, beforeAll, afterEach } from '@jest/globals';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithIntl } from '../../../common/utils/TestUtils.react18';
import ExperimentGenAIOverviewPage from './ExperimentGenAIOverviewPage';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MemoryRouter, Route, Routes } from '../../../common/utils/RoutingUtils';
import { MlflowService } from '../../sdk/MlflowService';
import { rest } from 'msw';
import { setupServer } from '../../../common/utils/setup-msw';
import { AggregationType, TraceMetricKey } from '@databricks/web-shared/model-trace-explorer';

// Mock MlflowService
jest.mock('../../sdk/MlflowService', () => ({
  MlflowService: {
    queryTraceMetrics: jest.fn(),
  },
}));

// Mock recharts to avoid rendering issues
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

const mockQueryTraceMetrics = MlflowService.queryTraceMetrics as jest.MockedFunction<
  typeof MlflowService.queryTraceMetrics
>;

describe('ExperimentGenAIOverviewPage', () => {
  const testExperimentId = 'test-experiment-456';

  const createQueryClient = () =>
    new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

  const renderComponent = (initialUrl = `/experiments/${testExperimentId}/overview`) => {
    const queryClient = createQueryClient();
    return renderWithIntl(
      <QueryClientProvider client={queryClient}>
        <DesignSystemProvider>
          <MemoryRouter initialEntries={[initialUrl]}>
            <Routes>
              <Route path="/experiments/:experimentId/overview" element={<ExperimentGenAIOverviewPage />} />
            </Routes>
          </MemoryRouter>
        </DesignSystemProvider>
      </QueryClientProvider>,
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
    // Default mock for queryTraceMetrics to return empty data
    mockQueryTraceMetrics.mockResolvedValue({
      data_points: [],
    });
  });

  describe('page rendering', () => {
    it('should render the Usage tab', async () => {
      renderComponent();

      await waitFor(() => {
        expect(screen.getByRole('tab', { name: 'Usage' })).toBeInTheDocument();
      });
    });

    it('should have Usage tab selected by default', async () => {
      renderComponent();

      await waitFor(() => {
        const usageTab = screen.getByRole('tab', { name: 'Usage' });
        expect(usageTab).toHaveAttribute('aria-selected', 'true');
      });
    });

    it('should render the tab content area', async () => {
      renderComponent();

      await waitFor(() => {
        expect(screen.getByRole('tabpanel')).toBeInTheDocument();
      });
    });

    it('should render the control bar with search and time range selector', async () => {
      renderComponent();

      await waitFor(() => {
        // Search input
        expect(screen.getByPlaceholderText('Search charts')).toBeInTheDocument();
        // Time range selector
        expect(screen.getByRole('combobox')).toBeInTheDocument();
      });
    });

    it('should have proper container structure', async () => {
      renderComponent();

      await waitFor(() => {
        // Check for tabs structure
        expect(screen.getByRole('tablist')).toBeInTheDocument();
        expect(screen.getByRole('tabpanel')).toBeInTheDocument();
      });
    });
  });

  describe('search input handling', () => {
    it('should render the search input with placeholder', async () => {
      renderComponent();

      await waitFor(() => {
        const searchInput = screen.getByPlaceholderText('Search charts');
        expect(searchInput).toBeInTheDocument();
      });
    });

    it('should allow typing in the search input', async () => {
      const user = userEvent.setup();
      renderComponent();

      await waitFor(() => {
        expect(screen.getByPlaceholderText('Search charts')).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText('Search charts');
      await user.type(searchInput, 'test query');

      expect(searchInput).toHaveValue('test query');
    });

    it('should update search query state when typing', async () => {
      const user = userEvent.setup();
      renderComponent();

      await waitFor(() => {
        expect(screen.getByPlaceholderText('Search charts')).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText('Search charts');
      await user.clear(searchInput);
      await user.type(searchInput, 'my search');

      expect(searchInput).toHaveValue('my search');
    });
  });

  describe('time range selection', () => {
    it('should render the date selector', async () => {
      renderComponent();

      await waitFor(() => {
        // Look for the date selector dropdown button
        const dateSelector = screen.getByRole('combobox');
        expect(dateSelector).toBeInTheDocument();
      });
    });

    it('should open time range dropdown when clicked', async () => {
      const user = userEvent.setup();
      renderComponent();

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toBeInTheDocument();
      });

      const dateSelector = screen.getByRole('combobox');
      await user.click(dateSelector);

      // Check that dropdown options are visible
      await waitFor(() => {
        expect(screen.getByRole('listbox')).toBeInTheDocument();
      });
    });

    it('should have a default time range selected', async () => {
      renderComponent();

      await waitFor(() => {
        const dateSelector = screen.getByRole('combobox');
        // Default is LAST_7_DAYS
        expect(dateSelector).toHaveTextContent(/7 days/i);
      });
    });

    it('should use LAST_7_DAYS as default time range for API calls', async () => {
      mockQueryTraceMetrics.mockResolvedValue({ data_points: [] });

      renderComponent();

      await waitFor(() => {
        expect(mockQueryTraceMetrics).toHaveBeenCalled();
      });

      // Verify that start_time_ms and end_time_ms are within expected range for 7 days
      const callArgs = mockQueryTraceMetrics.mock.calls[0][0];
      const now = Date.now();
      const sevenDaysAgo = now - 7 * 24 * 60 * 60 * 1000;

      expect(callArgs.start_time_ms).toBeGreaterThanOrEqual(sevenDaysAgo - 60000); // Allow 1 minute tolerance
      expect(callArgs.start_time_ms).toBeLessThanOrEqual(sevenDaysAgo + 60000);
    });

    it('should handle custom time range from URL parameters', async () => {
      const customStartTime = '2025-01-01T00:00:00.000Z';
      const customEndTime = '2025-01-07T23:59:59.999Z';
      const urlWithParams = `/experiments/${testExperimentId}/overview?startTimeLabel=CUSTOM&startTime=${encodeURIComponent(
        customStartTime,
      )}&endTime=${encodeURIComponent(customEndTime)}`;

      mockQueryTraceMetrics.mockResolvedValue({ data_points: [] });

      renderComponent(urlWithParams);

      await waitFor(() => {
        expect(mockQueryTraceMetrics).toHaveBeenCalledWith(
          expect.objectContaining({
            start_time_ms: expect.any(Number),
            end_time_ms: expect.any(Number),
          }),
        );
      });
    });
  });

  describe('chart integration', () => {
    it('should render TraceRequestsChart component', async () => {
      mockQueryTraceMetrics.mockResolvedValue({
        data_points: [
          {
            metric_name: TraceMetricKey.TRACE_COUNT,
            dimensions: { time_bucket: '2025-12-22T10:00:00Z' },
            values: { [AggregationType.COUNT]: 100 },
          },
        ],
      });

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Requests')).toBeInTheDocument();
      });
    });

    it('should display chart with data when API returns data', async () => {
      mockQueryTraceMetrics.mockResolvedValue({
        data_points: [
          {
            metric_name: TraceMetricKey.TRACE_COUNT,
            dimensions: { time_bucket: '2025-12-22T10:00:00Z' },
            values: { [AggregationType.COUNT]: 50 },
          },
          {
            metric_name: TraceMetricKey.TRACE_COUNT,
            dimensions: { time_bucket: '2025-12-22T11:00:00Z' },
            values: { [AggregationType.COUNT]: 75 },
          },
        ],
      });

      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      });

      // Verify total count is displayed (50 + 75 = 125)
      expect(screen.getByText('125')).toBeInTheDocument();
    });

    it('should show empty state when no data is available', async () => {
      mockQueryTraceMetrics.mockResolvedValue({
        data_points: [],
      });

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('No data available for the selected time range')).toBeInTheDocument();
      });
    });

    it('should show error state when API call fails', async () => {
      mockQueryTraceMetrics.mockRejectedValue(new Error('API Error'));

      renderComponent();

      await waitFor(() => {
        expect(screen.getByText('Failed to load chart data')).toBeInTheDocument();
      });
    });

    it('should pass experimentId to TraceRequestsChart', async () => {
      mockQueryTraceMetrics.mockResolvedValue({ data_points: [] });

      renderComponent();

      await waitFor(() => {
        expect(mockQueryTraceMetrics).toHaveBeenCalledWith(
          expect.objectContaining({
            experiment_ids: [testExperimentId],
          }),
        );
      });
    });
  });
});
