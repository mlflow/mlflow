import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { GatewayUsageSection } from './GatewayUsageSection';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';
import { setActiveWorkspace } from '../../../workspaces/utils/WorkspaceUtils';
import { getWorkspacesEnabledSync } from '../../../experiment-tracking/hooks/useServerInfo';

jest.mock('../../../experiment-tracking/hooks/useServerInfo', () => ({
  ...jest.requireActual<typeof import('../../../experiment-tracking/hooks/useServerInfo')>(
    '../../../experiment-tracking/hooks/useServerInfo',
  ),
  getWorkspacesEnabledSync: jest.fn(),
}));

const getWorkspacesEnabledSyncMock = jest.mocked(getWorkspacesEnabledSync);

// Mock GatewayChartsPanel
jest.mock('../GatewayChartsPanel', () => ({
  GatewayChartsPanel: () => (
    <div data-testid="gateway-charts-panel">
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

describe('GatewayUsageSection', () => {
  const testExperimentId = 'test-experiment-123';

  beforeEach(() => {
    jest.clearAllMocks();
    getWorkspacesEnabledSyncMock.mockReturnValue(false);
    setActiveWorkspace(null);
    if (typeof window !== 'undefined') {
      window.localStorage.clear();
    }
  });

  const renderComponent = (experimentId: string = testExperimentId) => {
    return renderWithDesignSystem(
      <MemoryRouter>
        <GatewayUsageSection experimentId={experimentId} />
      </MemoryRouter>,
    );
  };

  test('renders section title', () => {
    renderComponent();

    expect(screen.getByText('Usage')).toBeInTheDocument();
  });

  test('renders section description', () => {
    renderComponent();

    expect(screen.getByText('Monitor endpoint usage and performance metrics')).toBeInTheDocument();
  });

  test('renders view full dashboard link with correct experiment ID', () => {
    renderComponent();

    const link = screen.getByText('View full dashboard');
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute('href', `#/experiments/${testExperimentId}/overview`);
  });

  test('renders view full dashboard link with workspace param when workspace is active', () => {
    getWorkspacesEnabledSyncMock.mockReturnValue(true);
    setActiveWorkspace('my-workspace');
    renderComponent();

    const link = screen.getByText('View full dashboard');
    expect(link).toHaveAttribute('href', `#/experiments/${testExperimentId}/overview?workspace=my-workspace`);
  });

  test('renders view full dashboard link with workspace from localStorage when active workspace is missing', () => {
    getWorkspacesEnabledSyncMock.mockReturnValue(true);
    window.localStorage.setItem('mlflow.activeWorkspace', 'stored-workspace');
    renderComponent();

    const link = screen.getByText('View full dashboard');
    expect(link).toHaveAttribute('href', `#/experiments/${testExperimentId}/overview?workspace=stored-workspace`);
  });

  test('renders time range controls', () => {
    renderComponent();

    expect(screen.getByTestId('date-selector')).toBeInTheDocument();
    expect(screen.getByTestId('time-unit-selector')).toBeInTheDocument();
  });

  test('renders all chart components', () => {
    renderComponent();

    expect(screen.getByTestId('trace-requests-chart')).toBeInTheDocument();
    expect(screen.getByTestId('trace-latency-chart')).toBeInTheDocument();
    expect(screen.getByTestId('trace-errors-chart')).toBeInTheDocument();
    expect(screen.getByTestId('trace-token-usage-chart')).toBeInTheDocument();
    expect(screen.getByTestId('trace-token-stats-chart')).toBeInTheDocument();
    expect(screen.getByTestId('trace-cost-breakdown-chart')).toBeInTheDocument();
    expect(screen.getByTestId('trace-cost-over-time-chart')).toBeInTheDocument();
  });

  test('updates dashboard link when experiment ID changes', () => {
    const differentExperimentId = 'different-exp-456';
    renderComponent(differentExperimentId);

    const link = screen.getByText('View full dashboard');
    expect(link).toHaveAttribute('href', `#/experiments/${differentExperimentId}/overview`);
  });
});
