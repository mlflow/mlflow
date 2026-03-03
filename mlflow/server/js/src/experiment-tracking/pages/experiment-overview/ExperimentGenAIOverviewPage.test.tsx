import { jest, describe, it, expect, beforeEach, beforeAll, afterEach } from '@jest/globals';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithIntl } from '../../../common/utils/TestUtils.react18';
import ExperimentGenAIOverviewPage from './ExperimentGenAIOverviewPage';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

import { fetchOrFail } from '../../../common/utils/FetchUtils';
import { setupTestRouter, testRoute, TestRouter } from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';

// Mock FetchUtils
jest.mock('../../../common/utils/FetchUtils', () => ({
  fetchOrFail: jest.fn(),
  getAjaxUrl: (url: string) => url,
}));

// Mock useGetExperimentQuery to avoid Apollo client dependency
const mockUseGetExperimentQuery = jest.fn<() => { data: unknown; loading: boolean }>(() => ({
  data: undefined,
  loading: false,
}));
jest.mock('../../hooks/useExperimentQuery', () => ({
  useGetExperimentQuery: () => mockUseGetExperimentQuery(),
}));

// Mock useSearchMlflowTraces for demo experiment time range detection
const mockUseSearchMlflowTraces = jest.fn<() => { data: unknown[]; isLoading: boolean }>(() => ({
  data: [],
  isLoading: false,
}));
jest.mock('@databricks/web-shared/genai-traces-table', () => ({
  useSearchMlflowTraces: () => mockUseSearchMlflowTraces(),
  REQUEST_TIME_COLUMN_ID: 'request_time',
  TracesTableColumnType: { TRACE_INFO: 'TRACE_INFO' },
  invalidateMlflowSearchTracesCache: jest.fn(),
  SEARCH_MLFLOW_TRACES_QUERY_KEY: 'search-mlflow-traces',
}));

const mockFetchOrFail = jest.mocked(fetchOrFail);

// Demo experiment mock data (has mlflow.demo.version.* tag)
const createDemoExperimentMock = () => ({
  experimentId: 'test-experiment-456',
  name: 'Demo Experiment',
  artifactLocation: '/artifacts',
  creationTime: Date.now().toString(),
  lastUpdateTime: Date.now().toString(),
  lifecycleStage: 'active',
  tags: [{ key: 'mlflow.demo.version.traces', value: '1' }],
});

// Regular experiment mock data (no demo tags)
const createRegularExperimentMock = () => ({
  experimentId: 'test-experiment-456',
  name: 'Regular Experiment',
  artifactLocation: '/artifacts',
  creationTime: Date.now().toString(),
  lastUpdateTime: Date.now().toString(),
  lifecycleStage: 'active',
  tags: [],
});

describe('ExperimentGenAIOverviewPage', () => {
  const { history } = setupTestRouter();
  const testExperimentId = 'test-experiment-456';

  const createQueryClient = () =>
    new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

  const renderComponent = (initialUrl = `/experiments/${testExperimentId}/overview/usage`) => {
    const queryClient = createQueryClient();
    return renderWithIntl(
      <QueryClientProvider client={queryClient}>
        <DesignSystemProvider>
          <TestRouter
            history={history}
            routes={[testRoute(<ExperimentGenAIOverviewPage />, `/experiments/:experimentId/overview/:overviewTab?`)]}
            initialEntries={[initialUrl]}
          />
        </DesignSystemProvider>
      </QueryClientProvider>,
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
    // Default mock for fetchOrFail to return empty data
    mockFetchOrFail.mockResolvedValue({
      json: () => Promise.resolve({ data_points: [] }),
    } as Response);
    // Default mock for useGetExperimentQuery to return undefined (non-demo experiment)
    mockUseGetExperimentQuery.mockReturnValue({ data: undefined, loading: false });
    // Default mock for useSearchMlflowTraces to return empty data
    mockUseSearchMlflowTraces.mockReturnValue({ data: [], isLoading: false });
  });

  describe('page rendering', () => {
    it('should render all tabs', async () => {
      renderComponent();

      await waitFor(() => {
        expect(screen.getByRole('tab', { name: 'Usage' })).toBeInTheDocument();
        expect(screen.getByRole('tab', { name: 'Quality' })).toBeInTheDocument();
        expect(screen.getByRole('tab', { name: 'Tool calls' })).toBeInTheDocument();
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

    it('should render the control bar with time range and time unit selectors', async () => {
      renderComponent();

      await waitFor(() => {
        // Both time range and time unit selectors should be present
        const comboboxes = screen.getAllByRole('combobox');
        expect(comboboxes.length).toBeGreaterThanOrEqual(2);
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

    it('should switch to Quality tab when clicked', async () => {
      const user = userEvent.setup();
      renderComponent();

      await waitFor(() => {
        expect(screen.getByRole('tab', { name: 'Quality' })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('tab', { name: 'Quality' }));

      await waitFor(() => {
        const qualityTab = screen.getByRole('tab', { name: 'Quality' });
        expect(qualityTab).toHaveAttribute('aria-selected', 'true');
      });
    });

    it('should switch to Tool calls tab when clicked', async () => {
      const user = userEvent.setup();
      renderComponent();

      await waitFor(() => {
        expect(screen.getByRole('tab', { name: 'Tool calls' })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('tab', { name: 'Tool calls' }));

      await waitFor(() => {
        const toolCallsTab = screen.getByRole('tab', { name: 'Tool calls' });
        expect(toolCallsTab).toHaveAttribute('aria-selected', 'true');
      });
    });
  });

  describe('time range selection', () => {
    it('should render the date selector', async () => {
      renderComponent();

      await waitFor(() => {
        // Look for the date selector by its test id
        const dateSelector = screen.getByTestId('time-range-select-dropdown');
        expect(dateSelector).toBeInTheDocument();
      });
    });

    it('should open time range dropdown when clicked', async () => {
      const user = userEvent.setup();
      renderComponent();

      await waitFor(() => {
        expect(screen.getByTestId('time-range-select-dropdown')).toBeInTheDocument();
      });

      const dateSelector = screen.getByTestId('time-range-select-dropdown');
      await user.click(dateSelector);

      // Check that dropdown options are visible
      await waitFor(() => {
        expect(screen.getByRole('listbox')).toBeInTheDocument();
      });
    });

    it('should have a default time range selected', async () => {
      renderComponent();

      await waitFor(() => {
        const dateSelector = screen.getByTestId('time-range-select-dropdown');
        // Default is LAST_7_DAYS
        expect(dateSelector).toHaveTextContent(/7 days/i);
      });
    });

    it('should use LAST_7_DAYS as default time range for API calls', async () => {
      renderComponent();

      await waitFor(() => {
        expect(mockFetchOrFail).toHaveBeenCalled();
      });

      // Verify that start_time_ms and end_time_ms are within expected range for 7 days
      const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
      const now = Date.now();
      const sevenDaysAgo = now - 7 * 24 * 60 * 60 * 1000;

      expect(callBody.start_time_ms).toBeGreaterThanOrEqual(sevenDaysAgo - 60000); // Allow 1 minute tolerance
      expect(callBody.start_time_ms).toBeLessThanOrEqual(sevenDaysAgo + 60000);
    });

    it('should handle custom time range from URL parameters', async () => {
      const customStartTime = '2025-01-01T00:00:00.000Z';
      const customEndTime = '2025-01-07T23:59:59.999Z';
      const urlWithParams = `/experiments/${testExperimentId}/overview/usage?startTimeLabel=CUSTOM&startTime=${encodeURIComponent(
        customStartTime,
      )}&endTime=${encodeURIComponent(customEndTime)}`;

      renderComponent(urlWithParams);

      await waitFor(() => {
        const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
        expect(callBody.start_time_ms).toEqual(expect.any(Number));
        expect(callBody.end_time_ms).toEqual(expect.any(Number));
      });
    });

    it('should set custom time range based on actual trace data for demo experiments', async () => {
      // Mock trace data with specific timestamps (oldest trace 10 days ago, newest 5 days ago)
      const tenDaysAgo = new Date(Date.now() - 10 * 24 * 60 * 60 * 1000).toISOString();
      const fiveDaysAgo = new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString();

      const mockTraces = [
        { request_time: fiveDaysAgo, trace_id: 'trace-1' }, // newest (first in desc order)
        { request_time: tenDaysAgo, trace_id: 'trace-2' }, // oldest (last in desc order)
      ];

      // Mock to return a demo experiment
      mockUseGetExperimentQuery.mockReturnValue({
        data: createDemoExperimentMock(),
        loading: false,
      });

      // Mock to return traces with specific timestamps
      mockUseSearchMlflowTraces.mockReturnValue({
        data: mockTraces,
        isLoading: false,
      });

      renderComponent();

      // For demo experiments, the time range should be updated to a custom range
      // based on actual trace timestamps
      await waitFor(() => {
        const dateSelector = screen.getByTestId('time-range-select-dropdown');
        expect(dateSelector).toHaveTextContent(/Custom/i);
      });

      // Verify that API calls use the custom time range based on trace data
      await waitFor(() => {
        const calls = mockFetchOrFail.mock.calls;
        expect(calls.length).toBeGreaterThan(0);

        // Get the last call which should have the custom range
        const lastCallBody = JSON.parse((calls[calls.length - 1]?.[1] as any)?.body || '{}');

        // The start time should be close to the oldest trace (10 days ago)
        const tenDaysAgoMs = Date.now() - 10 * 24 * 60 * 60 * 1000;
        expect(lastCallBody.start_time_ms).toBeGreaterThanOrEqual(tenDaysAgoMs - 60000);
        expect(lastCallBody.start_time_ms).toBeLessThanOrEqual(tenDaysAgoMs + 60000);
      });
    });

    it('should respect explicit URL time range even for demo experiments', async () => {
      // Mock to return a demo experiment
      mockUseGetExperimentQuery.mockReturnValue({
        data: createDemoExperimentMock(),
        loading: false,
      });

      // Explicitly set time range to LAST_7_DAYS in URL
      const urlWithParams = `/experiments/${testExperimentId}/overview/usage?startTimeLabel=LAST_7_DAYS`;

      renderComponent(urlWithParams);

      // Even for demo experiments, explicit URL param should be respected
      await waitFor(() => {
        const dateSelector = screen.getByTestId('time-range-select-dropdown');
        expect(dateSelector).toHaveTextContent(/7 days/i);
      });
    });
  });

  describe('chart integration', () => {
    it('should render all chart components', async () => {
      renderComponent();

      // Verify all charts are rendered
      await waitFor(() => {
        expect(screen.getByText('Traces')).toBeInTheDocument();
        expect(screen.getByText('Latency')).toBeInTheDocument();
        expect(screen.getByText('Errors')).toBeInTheDocument();
      });
    });

    it('should pass experimentId to chart API calls', async () => {
      renderComponent();

      await waitFor(() => {
        expect(mockFetchOrFail).toHaveBeenCalled();
      });

      // Verify experimentId is passed to chart API calls
      const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
      expect(callBody.experiment_ids).toEqual([testExperimentId]);
    });

    it('should render charts in correct layout', async () => {
      renderComponent();

      await waitFor(() => {
        // Traces chart should be present (full width)
        expect(screen.getByText('Traces')).toBeInTheDocument();
        // Latency and Errors charts should be present (side by side)
        expect(screen.getByText('Latency')).toBeInTheDocument();
        expect(screen.getByText('Errors')).toBeInTheDocument();
      });
    });
  });
});
