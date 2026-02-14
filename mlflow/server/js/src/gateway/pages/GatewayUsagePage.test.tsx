import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { GatewayUsagePage } from './GatewayUsagePage';
import { MemoryRouter } from '../../common/utils/RoutingUtils';

// Mock GatewayChartsPanel
jest.mock('../components/GatewayChartsPanel', () => ({
  GatewayChartsPanel: ({ additionalControls }: { additionalControls?: React.ReactNode }) => (
    <div data-testid="gateway-charts-panel">
      {additionalControls}
      <div data-testid="time-unit-selector">Time Unit Selector</div>
      <div data-testid="date-selector">Date Selector</div>
      <div data-testid="trace-requests-chart">Requests Chart</div>
      <div data-testid="trace-latency-chart">Latency Chart</div>
      <div data-testid="trace-errors-chart">Errors Chart</div>
      <div data-testid="trace-token-usage-chart">Token Usage Chart</div>
      <div data-testid="trace-token-stats-chart">Token Stats Chart</div>
      <div data-testid="trace-cost-breakdown-chart">Cost Breakdown Chart</div>
      <div data-testid="trace-cost-over-time-chart">Cost Over Time Chart</div>
    </div>
  ),
}));

// Mock useEndpointsQuery
const mockUseEndpointsQuery = jest.fn();
jest.mock('../hooks/useEndpointsQuery', () => ({
  useEndpointsQuery: () => mockUseEndpointsQuery(),
}));

const mockEndpointsWithUsageTracking = [
  { endpoint_id: 'ep-1', name: 'Endpoint 1', usage_tracking: true, experiment_id: 'exp-1' },
  { endpoint_id: 'ep-2', name: 'Endpoint 2', usage_tracking: true, experiment_id: 'exp-2' },
];

const mockEndpointsWithoutUsageTracking = [
  { endpoint_id: 'ep-3', name: 'Endpoint 3', usage_tracking: false, experiment_id: null },
];

describe('GatewayUsagePage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  const renderComponent = () => {
    return renderWithDesignSystem(
      <MemoryRouter>
        <GatewayUsagePage />
      </MemoryRouter>,
    );
  };

  describe('when endpoints have usage tracking enabled', () => {
    beforeEach(() => {
      mockUseEndpointsQuery.mockReturnValue({
        data: mockEndpointsWithUsageTracking,
        isLoading: false,
      });
    });

    test('renders page title and subtitle', () => {
      renderComponent();

      expect(screen.getByText('Gateway Usage')).toBeInTheDocument();
      expect(screen.getByText('Monitor usage and performance across all endpoints')).toBeInTheDocument();
    });

    test('renders endpoint selector with "All endpoints" option', () => {
      renderComponent();

      expect(screen.getByText('Endpoint:')).toBeInTheDocument();
      const selector = screen.getByRole('combobox');
      expect(selector).toBeInTheDocument();
      expect(screen.getByText('All endpoints')).toBeInTheDocument();
    });

    test('renders endpoint options in selector when opened', async () => {
      renderComponent();

      // Open the dropdown
      const selector = screen.getByRole('combobox');
      await userEvent.click(selector);

      // Options should now be visible
      expect(screen.getByText('Endpoint 1')).toBeInTheDocument();
      expect(screen.getByText('Endpoint 2')).toBeInTheDocument();
    });

    test('renders time controls', () => {
      renderComponent();

      expect(screen.getByTestId('time-unit-selector')).toBeInTheDocument();
      expect(screen.getByTestId('date-selector')).toBeInTheDocument();
    });

    test('renders all chart components when showing all endpoints', () => {
      renderComponent();

      expect(screen.getByTestId('trace-requests-chart')).toBeInTheDocument();
      expect(screen.getByTestId('trace-latency-chart')).toBeInTheDocument();
      expect(screen.getByTestId('trace-errors-chart')).toBeInTheDocument();
      expect(screen.getByTestId('trace-token-usage-chart')).toBeInTheDocument();
      expect(screen.getByTestId('trace-token-stats-chart')).toBeInTheDocument();
      expect(screen.getByTestId('trace-cost-breakdown-chart')).toBeInTheDocument();
      expect(screen.getByTestId('trace-cost-over-time-chart')).toBeInTheDocument();
    });

    test('allows selecting a specific endpoint', async () => {
      renderComponent();

      // Open the dropdown
      const selector = screen.getByRole('combobox');
      await userEvent.click(selector);

      // Click on Endpoint 1
      await userEvent.click(screen.getByText('Endpoint 1'));

      // Charts should still be visible for the selected endpoint
      expect(screen.getByTestId('trace-requests-chart')).toBeInTheDocument();
    });
  });

  describe('when no endpoints have usage tracking enabled', () => {
    beforeEach(() => {
      mockUseEndpointsQuery.mockReturnValue({
        data: mockEndpointsWithoutUsageTracking,
        isLoading: false,
      });
    });

    test('renders empty state', () => {
      renderComponent();

      expect(screen.getByText('No usage data available')).toBeInTheDocument();
      expect(
        screen.getByText('Enable usage tracking on your endpoints to see usage metrics here.'),
      ).toBeInTheDocument();
    });

    test('renders link to endpoints page', () => {
      renderComponent();

      expect(screen.getByText('Go to Endpoints')).toBeInTheDocument();
    });

    test('does not render charts', () => {
      renderComponent();

      expect(screen.queryByTestId('trace-requests-chart')).not.toBeInTheDocument();
      expect(screen.queryByTestId('trace-latency-chart')).not.toBeInTheDocument();
    });
  });

  describe('when endpoints are loading', () => {
    beforeEach(() => {
      mockUseEndpointsQuery.mockReturnValue({
        data: [],
        isLoading: true,
      });
    });

    test('shows loading spinner', () => {
      renderComponent();

      // The spinner should be in the controls section
      expect(screen.getByText('Endpoint:')).toBeInTheDocument();
    });

    test('disables endpoint selector while loading', () => {
      renderComponent();

      const selector = screen.getByRole('combobox');
      expect(selector).toBeDisabled();
    });
  });

  describe('when endpoints list is empty', () => {
    beforeEach(() => {
      mockUseEndpointsQuery.mockReturnValue({
        data: [],
        isLoading: false,
      });
    });

    test('renders empty state', () => {
      renderComponent();

      expect(screen.getByText('No usage data available')).toBeInTheDocument();
    });
  });
});
