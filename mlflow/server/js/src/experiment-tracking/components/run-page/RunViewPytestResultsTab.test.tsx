import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { renderWithIntl, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { MemoryRouter } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { DesignSystemProvider } from '@databricks/design-system';
import { RunViewPytestResultsTab } from './RunViewPytestResultsTab';
import { MlflowService } from '../../sdk/MlflowService';

jest.mock('../../sdk/MlflowService', () => ({
  MlflowService: {
    searchRuns: jest.fn(),
  },
}));

jest.mock('../../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/RoutingUtils')>('../../../common/utils/RoutingUtils'),
  useParams: () => ({ experimentId: 'exp-1', runUuid: 'parent-run' }),
}));

const parentRunUuid = 'parent-run';

const createMockRun = (
  runId: string,
  runName: string,
  outcome: string,
  duration: string,
  metrics: Array<{ key: string; value: number }> = [],
) => ({
  info: { run_uuid: runId, run_id: runId, run_name: runName },
  data: {
    tags: [
      { key: 'mlflow.test.outcome', value: outcome },
      { key: 'mlflow.test.duration', value: duration },
    ],
    metrics,
    params: [],
  },
});

describe('RunViewPytestResultsTab', () => {
  const renderComponent = () =>
    renderWithIntl(
      <MemoryRouter>
        <DesignSystemProvider>
          <RunViewPytestResultsTab runUuid={parentRunUuid} />
        </DesignSystemProvider>
      </MemoryRouter>,
    );

  beforeEach(() => {
    jest.mocked(MlflowService.searchRuns).mockReset();
  });

  test('displays loading state then renders test results', async () => {
    jest.mocked(MlflowService.searchRuns).mockResolvedValueOnce({
      runs: [
        createMockRun('run-1', 'test_llm_accuracy', 'passed', '1.234'),
        createMockRun('run-2', 'test_llm_safety', 'failed', '2.567'),
        createMockRun('run-3', 'test_llm_speed', 'skipped', '0.100'),
      ],
    } as any);

    renderComponent();

    // Should show loading initially
    expect(screen.getByText('Loading test results...')).toBeInTheDocument();

    // Should render results
    await waitFor(() => {
      expect(screen.getByTestId('pytest-results-table')).toBeInTheDocument();
    });

    // Check summary
    const summary = screen.getByTestId('pytest-results-summary');
    expect(summary).toHaveTextContent('1 passed');
    expect(summary).toHaveTextContent('1 failed');
    expect(summary).toHaveTextContent('1 skipped');

    // Check test names rendered
    expect(screen.getByText('test_llm_accuracy')).toBeInTheDocument();
    expect(screen.getByText('test_llm_safety')).toBeInTheDocument();
    expect(screen.getByText('test_llm_speed')).toBeInTheDocument();

    // Check outcome badges
    expect(screen.getByText('passed')).toBeInTheDocument();
    expect(screen.getByText('failed')).toBeInTheDocument();
    expect(screen.getByText('skipped')).toBeInTheDocument();
  });

  test('displays empty state when no test results', async () => {
    jest.mocked(MlflowService.searchRuns).mockResolvedValueOnce({ runs: [] } as any);

    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('No test results found.')).toBeInTheDocument();
    });
  });

  test('displays error state on API failure', async () => {
    jest.mocked(MlflowService.searchRuns).mockRejectedValueOnce(new Error('Network error'));

    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('Failed to load test results.')).toBeInTheDocument();
    });
  });

  test('renders metrics in the table', async () => {
    jest.mocked(MlflowService.searchRuns).mockResolvedValueOnce({
      runs: [
        createMockRun('run-1', 'test_with_metrics', 'passed', '1.000', [
          { key: 'answer_similarity/mean', value: 0.95 },
        ]),
      ],
    } as any);

    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('test_with_metrics')).toBeInTheDocument();
    });

    expect(screen.getByText('answer_similarity/mean=0.950')).toBeInTheDocument();
  });

  test('calls searchRuns with correct filter', async () => {
    jest.mocked(MlflowService.searchRuns).mockResolvedValueOnce({ runs: [] } as any);

    renderComponent();

    await waitFor(() => {
      expect(MlflowService.searchRuns).toHaveBeenCalledWith(
        expect.objectContaining({
          experiment_ids: ['exp-1'],
          filter: expect.stringContaining(parentRunUuid),
        }),
      );
    });
  });
});
