import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { GatewayUsageSection } from './GatewayUsageSection';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';

// Mock the lazy chart components
jest.mock('../../../experiment-tracking/pages/experiment-overview/components/LazyTraceRequestsChart', () => ({
  LazyTraceRequestsChart: () => <div data-testid="trace-requests-chart">Requests Chart</div>,
}));

jest.mock('../../../experiment-tracking/pages/experiment-overview/components/LazyTraceLatencyChart', () => ({
  LazyTraceLatencyChart: () => <div data-testid="trace-latency-chart">Latency Chart</div>,
}));

jest.mock('../../../experiment-tracking/pages/experiment-overview/components/LazyTraceErrorsChart', () => ({
  LazyTraceErrorsChart: () => <div data-testid="trace-errors-chart">Errors Chart</div>,
}));

jest.mock('../../../experiment-tracking/pages/experiment-overview/components/LazyTraceTokenUsageChart', () => ({
  LazyTraceTokenUsageChart: () => <div data-testid="trace-token-usage-chart">Token Usage Chart</div>,
}));

jest.mock('../../../experiment-tracking/pages/experiment-overview/components/LazyTraceTokenStatsChart', () => ({
  LazyTraceTokenStatsChart: () => <div data-testid="trace-token-stats-chart">Token Stats Chart</div>,
}));

jest.mock('../../../experiment-tracking/pages/experiment-overview/components/LazyTraceCostBreakdownChart', () => ({
  LazyTraceCostBreakdownChart: () => <div data-testid="trace-cost-breakdown-chart">Cost Breakdown Chart</div>,
}));

jest.mock('../../../experiment-tracking/pages/experiment-overview/components/LazyTraceCostOverTimeChart', () => ({
  LazyTraceCostOverTimeChart: () => <div data-testid="trace-cost-over-time-chart">Cost Over Time Chart</div>,
}));

// Mock TracesV3DateSelector
jest.mock('../../../experiment-tracking/components/experiment-page/components/traces-v3/TracesV3DateSelector', () => ({
  TracesV3DateSelector: () => <div data-testid="date-selector">Date Selector</div>,
}));

// Mock TimeUnitSelector
jest.mock('../../../experiment-tracking/pages/experiment-overview/components/TimeUnitSelector', () => ({
  TimeUnitSelector: () => <div data-testid="time-unit-selector">Time Unit Selector</div>,
}));

describe('GatewayUsageSection', () => {
  const testExperimentId = 'test-experiment-123';

  beforeEach(() => {
    jest.clearAllMocks();
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
