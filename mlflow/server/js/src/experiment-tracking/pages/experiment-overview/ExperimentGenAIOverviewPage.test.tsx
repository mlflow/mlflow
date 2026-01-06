import { jest, describe, it, expect, beforeEach, beforeAll, afterEach } from '@jest/globals';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithIntl } from '../../../common/utils/TestUtils.react18';
import ExperimentGenAIOverviewPage from './ExperimentGenAIOverviewPage';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MemoryRouter, Route, Routes } from '../../../common/utils/RoutingUtils';

// Mock FetchUtils
jest.mock('../../../common/utils/FetchUtils', () => ({
  fetchOrFail: jest.fn(),
  getAjaxUrl: (url: string) => url,
}));

import { fetchOrFail } from '../../../common/utils/FetchUtils';
const mockFetchOrFail = fetchOrFail as jest.MockedFunction<typeof fetchOrFail>;

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
    // Default mock for fetchOrFail to return empty data
    mockFetchOrFail.mockResolvedValue({
      json: () => Promise.resolve({ data_points: [] }),
    } as Response);
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
      const urlWithParams = `/experiments/${testExperimentId}/overview?startTimeLabel=CUSTOM&startTime=${encodeURIComponent(
        customStartTime,
      )}&endTime=${encodeURIComponent(customEndTime)}`;

      renderComponent(urlWithParams);

      await waitFor(() => {
        const callBody = JSON.parse((mockFetchOrFail.mock.calls[0]?.[1] as any)?.body || '{}');
        expect(callBody.start_time_ms).toEqual(expect.any(Number));
        expect(callBody.end_time_ms).toEqual(expect.any(Number));
      });
    });
  });

  describe('chart integration', () => {
    it('should render all chart components', async () => {
      renderComponent();

      // Verify all charts are rendered
      await waitFor(() => {
        expect(screen.getByText('Requests')).toBeInTheDocument();
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
        // Requests chart should be present (full width)
        expect(screen.getByText('Requests')).toBeInTheDocument();
        // Latency and Errors charts should be present (side by side)
        expect(screen.getByText('Latency')).toBeInTheDocument();
        expect(screen.getByText('Errors')).toBeInTheDocument();
      });
    });
  });
});
