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

  test('preserves all query params when navigating between traces-related tabs', async () => {
    renderTestComponent([
      '/experiments/test-123/traces?startTimeLabel=LAST_24_HOURS&startTime=2024-01-01&selectedTraceId=trace-123',
    ]);

    // Sessions link (traces-related tab) - should preserve ALL params
    const sessionsLink = screen.getByText('Sessions').closest('a');
    expect(sessionsLink).toHaveAttribute('href', expect.stringContaining('startTimeLabel=LAST_24_HOURS'));
    expect(sessionsLink).toHaveAttribute('href', expect.stringContaining('startTime=2024-01-01'));
    expect(sessionsLink).toHaveAttribute('href', expect.stringContaining('selectedTraceId=trace-123'));

    // Overview link (traces-related tab) - should preserve ALL params
    const overviewLink = screen.getByText('Overview').closest('a');
    expect(overviewLink).toHaveAttribute('href', expect.stringContaining('startTimeLabel=LAST_24_HOURS'));
    expect(overviewLink).toHaveAttribute('href', expect.stringContaining('selectedTraceId=trace-123'));
  });

  test('preserves only time range params when navigating from traces-related to other tabs', () => {
    renderTestComponent(['/experiments/test-123/traces?startTimeLabel=LAST_24_HOURS&selectedTraceId=trace-123']);

    // Datasets link - should only preserve time range params
    const datasetsLink = screen.getByText('Datasets').closest('a');
    expect(datasetsLink).toHaveAttribute('href', expect.stringContaining('startTimeLabel=LAST_24_HOURS'));
    expect(datasetsLink).not.toHaveAttribute('href', expect.stringContaining('selectedTraceId'));
  });

  test('preserves only time range params when navigating from other tabs to traces-related', () => {
    renderTestComponent(['/experiments/test-123/datasets?startTimeLabel=LAST_24_HOURS&selectedDatasetId=dataset-123']);

    // Traces link - should only preserve time range params
    const tracesLink = screen.getByText('Traces').closest('a');
    expect(tracesLink).toHaveAttribute('href', expect.stringContaining('startTimeLabel=LAST_24_HOURS'));
    expect(tracesLink).not.toHaveAttribute('href', expect.stringContaining('selectedDatasetId'));
  });

  test('preserves only time range params when navigating between non-traces-related tabs', () => {
    renderTestComponent([
      '/experiments/test-123/datasets?startTimeLabel=LAST_24_HOURS&selectedDatasetId=dataset-123&viewMode=list',
    ]);

    // Judges link - should only preserve time range params
    const judgesLink = screen.getByText('Judges').closest('a');
    expect(judgesLink).toHaveAttribute('href', expect.stringContaining('startTimeLabel=LAST_24_HOURS'));
    expect(judgesLink).not.toHaveAttribute('href', expect.stringContaining('selectedDatasetId'));
    expect(judgesLink).not.toHaveAttribute('href', expect.stringContaining('viewMode'));
  });
});
