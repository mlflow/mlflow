import { describe, jest, test, expect, beforeEach } from '@jest/globals';
import { screen } from '@testing-library/react';
import { MlflowSidebarExperimentItems } from './MlflowSidebarExperimentItems';
import { MemoryRouter } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { QueryClient, QueryClientProvider } from '../../common/utils/reactQueryHooks';
import { WorkflowType } from '../contexts/WorkflowTypeContext';
import { renderWithDesignSystem } from '../utils/TestUtils.react18';

jest.mock('../../experiment-tracking/hooks/useExperimentQuery', () => ({
  useGetExperimentQuery: jest.fn(() => ({})),
}));

jest.mock('../../experiment-tracking/components/experiment-page/hooks/useExperimentEvaluationRunsData', () => ({
  useExperimentEvaluationRunsData: jest.fn(() => ({ trainingRuns: [] })),
}));

describe('MlflowSidebarExperimentItems', () => {
  const renderTestComponent = (initialEntries: string[] = ['/experiments/test-123/traces']) => {
    const queryClient = new QueryClient();
    return renderWithDesignSystem(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={initialEntries}>
          <MlflowSidebarExperimentItems collapsed={false} experimentId="test-123" workflowType={WorkflowType.GENAI} />
        </MemoryRouter>
      </QueryClientProvider>,
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('preserves query params when navigating between traces-related tabs', async () => {
    renderTestComponent(['/experiments/test-123/traces?startTimeLabel=LAST_24_HOURS&startTime=2024-01-01']);

    // Find the Sessions link (traces-related tab)
    const sessionsLink = screen.getByText('Sessions').closest('a');
    expect(sessionsLink).toHaveAttribute('href', expect.stringContaining('startTimeLabel=LAST_24_HOURS'));
    expect(sessionsLink).toHaveAttribute('href', expect.stringContaining('startTime=2024-01-01'));

    // Find the Overview link (traces-related tab)
    const overviewLink = screen.getByText('Overview').closest('a');
    expect(overviewLink).toHaveAttribute('href', expect.stringContaining('startTimeLabel=LAST_24_HOURS'));
  });

  test('preserves query params when navigating to evaluation tabs', () => {
    renderTestComponent(['/experiments/test-123/traces?startTimeLabel=LAST_24_HOURS']);

    // Find the Datasets link (now a traces-related tab)
    const datasetsLink = screen.getByText('Datasets').closest('a');
    expect(datasetsLink).toHaveAttribute('href', expect.stringContaining('startTimeLabel=LAST_24_HOURS'));
  });

  test('preserves query params when navigating from evaluation tabs', () => {
    renderTestComponent(['/experiments/test-123/datasets?startTimeLabel=LAST_24_HOURS']);

    // Query params should be preserved when navigating from Datasets to Traces
    const tracesLink = screen.getByText('Traces').closest('a');
    expect(tracesLink).toHaveAttribute('href', expect.stringContaining('startTimeLabel=LAST_24_HOURS'));
  });
});
