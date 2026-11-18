import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { setupTestRouter, testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import { shouldEnableNewEvalRunsUi } from '../../../common/utils/FeatureUtils';
import { MlflowRunStatus } from '../../../graphql/__generated__/graphql';
import EvalRunDetailsPage, { getEvalRunScores } from './EvalRunDetailsPage';
import { useEvalRunProgress } from './useEvalRunProgress';
import { useGetExperimentQuery } from '../../hooks/useExperimentQuery';
import {
  ExperimentPageRevampProvider,
  useExperimentBreadcrumbPairwiseReviewAction,
} from '../experiment-page-tabs/ExperimentPageRevampContext';

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableNewEvalRunsUi: jest.fn(() => false),
}));
jest.mock('./useEvalRunProgress', () => ({ useEvalRunProgress: jest.fn() }));
jest.mock('../../hooks/useExperimentQuery', () => ({ useGetExperimentQuery: jest.fn() }));
jest.mock('../experiment-evaluation-runs/hooks/useEvalRunsBaseline', () => ({
  ...jest.requireActual<typeof import('../experiment-evaluation-runs/hooks/useEvalRunsBaseline')>(
    '../experiment-evaluation-runs/hooks/useEvalRunsBaseline',
  ),
  useGetEvalRunsBaseline: (runUuid?: string) => ({
    data: runUuid
      ? {
          info: { runUuid, experimentId: '123', lifecycleStage: 'active', runName: 'Baseline run' },
          data: { metrics: [{ key: 'safety/mean', value: 0.5 }], params: [], tags: [] },
        }
      : undefined,
  }),
}));
jest.mock('../../components/evaluations/RunViewEvaluationsTab', () => ({
  RunViewEvaluationsTab: ({ runTags }: { runTags?: Record<string, unknown> }) => (
    <div data-testid="eval-results-table">run-tags:{runTags ? 'present' : 'absent'}</div>
  ),
}));
// Keep this routing test lightweight; the sidebar and the table have their own tests.
jest.mock('./EvalRunTracesSidebar', () => ({
  EvalRunTracesSidebar: () => <div role="region" aria-label="Run sidebar" />,
}));
const mockStartReview = jest.fn();
jest.mock('./EvalRunCompareView', () => {
  const { useEffect: useMockEffect } = jest.requireActual<typeof import('react')>('react');
  return {
    EvalRunCompareView: ({
      baselineRunUuid,
      compareRunUuids,
      onPairwiseReviewActionChange,
    }: {
      baselineRunUuid: string;
      compareRunUuids: string[];
      onPairwiseReviewActionChange?: (action: (() => void) | undefined) => void;
    }) => {
      useMockEffect(() => {
        onPairwiseReviewActionChange?.(mockStartReview);
        return () => onPairwiseReviewActionChange?.(undefined);
      }, [onPairwiseReviewActionChange]);
      return (
        <div data-testid="eval-run-compare-view">
          base:{baselineRunUuid} compare:{compareRunUuids.join(',')}
        </div>
      );
    },
  };
});
// Reads the breadcrumb's published "start pairwise review" action, as ExperimentPageBreadcrumbs does.
const ReviewActionProbe = () => {
  const action = useExperimentBreadcrumbPairwiseReviewAction();
  return action ? (
    <button type="button" onClick={action}>
      Published review
    </button>
  ) : null;
};
jest.mock('./EvalRunTracesV4Table', () => ({
  EvalRunTracesV4Table: ({
    runUuid,
    baselineRunUuid,
    currentScores,
    baselineScores,
  }: {
    runUuid: string;
    baselineRunUuid?: string;
    currentScores?: Record<string, number>;
    baselineScores?: Record<string, number>;
  }) => (
    <div role="table" aria-label="Evaluation run traces" data-testid="eval-results-v4">
      run:{runUuid} baseline:{baselineRunUuid ?? 'none'} currentSafety:{currentScores?.['safety'] ?? 'none'}{' '}
      baselineSafety:{baselineScores?.['safety'] ?? 'none'}
    </div>
  ),
}));
jest.mock('../../components/experiment-page/components/traces-v4/components/TracesV4PageWrapper', () => ({
  TracesV4PageWrapper: ({ children }: { children: React.ReactNode }) => children,
}));
jest.mock('../../components/experiment-page/components/traces-v3/ExperimentCustomViewProvider', () => ({
  ExperimentCustomViewProvider: ({ experimentId, children }: { experimentId?: string; children: React.ReactNode }) => (
    <div data-testid="custom-view-provider" data-experiment-id={experimentId}>
      {children}
    </div>
  ),
}));

type Progress = ReturnType<typeof useEvalRunProgress>;
const baseProgress: Progress = {
  isLoading: false,
  isNotFound: false,
  loadError: undefined,
  status: MlflowRunStatus.RUNNING,
  runName: 'my-eval-run',
  experiment: undefined,
  runMetrics: [],
  runTags: {},
  isFinished: false,
};

const mockBaselineTag = (value?: string) =>
  jest.mocked(useGetExperimentQuery).mockReturnValue({
    data: value ? { tags: [{ key: 'mlflow.evaluation.baselineRunId', value }] } : undefined,
  } as unknown as ReturnType<typeof useGetExperimentQuery>);

describe('EvalRunDetailsPage', () => {
  const { history } = setupTestRouter();

  beforeEach(() => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    jest.mocked(useEvalRunProgress).mockReturnValue(baseProgress);
    mockBaselineTag(undefined);
  });

  const renderAt = (entry: string, revampEnabled = false) => {
    render(<div />, {
      wrapper: () => (
        <IntlProvider locale="en">
          <DesignSystemProvider>
            <TestRouter
              history={history}
              routes={[
                testRoute(
                  <ExperimentPageRevampProvider enabled={revampEnabled}>
                    <EvalRunDetailsPage />
                    <ReviewActionProbe />
                  </ExperimentPageRevampProvider>,
                  '/experiments/:experimentId/evaluation-runs/:runUuid/details',
                ),
                testRoute(<div>Experiment page</div>, '/experiments/:experimentId'),
                testRoute(<div />, '*'),
              ]}
              initialEntries={[entry]}
            />
          </DesignSystemProvider>
        </IntlProvider>
      ),
    });
  };

  test('redirects to the experiment page when the flag is off', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
    renderAt('/experiments/123/evaluation-runs/run-1/details');
    expect(await screen.findByText('Experiment page')).toBeInTheDocument();
    expect(screen.queryByTestId('eval-results-v4')).not.toBeInTheDocument();
  });

  test('opens the run-scoped traces table with a sidebar and no single-run tabs', async () => {
    jest.mocked(useEvalRunProgress).mockReturnValue({
      ...baseProgress,
      status: MlflowRunStatus.FINISHED,
      isFinished: true,
    });
    renderAt('/experiments/123/evaluation-runs/run-1/details');

    expect(await screen.findByTestId('eval-results-v4')).toHaveTextContent('run:run-1');
    expect(screen.getByRole('region', { name: 'Run sidebar' })).toBeInTheDocument();
    expect(screen.queryByTestId('eval-results-table')).not.toBeInTheDocument();
    expect(screen.queryByRole('tab', { name: 'Overview' })).not.toBeInTheDocument();
    expect(screen.queryByRole('tab', { name: 'Traces' })).not.toBeInTheDocument();
  });

  test('opens the comparison of the path run against the compared runs in compare mode', async () => {
    renderAt('/experiments/123/evaluation-runs/run-1/details?compare=true&compareFrom=runs&compareRuns=run-2,run-3');

    expect(await screen.findByTestId('eval-run-compare-view')).toHaveTextContent('base:run-1 compare:run-2,run-3');
    expect(screen.queryByTestId('eval-results-v4')).not.toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Run sidebar' })).not.toBeInTheDocument();
  });

  test('publishes the compare view’s pairwise review action for the breadcrumb while comparing', async () => {
    mockStartReview.mockClear();
    renderAt('/experiments/123/evaluation-runs/run-1/details?compare=true&compareFrom=runs&compareRuns=run-2', true);
    await userEvent.click(await screen.findByRole('button', { name: 'Published review' }));
    expect(mockStartReview).toHaveBeenCalledTimes(1);
  });

  test('publishes no pairwise review action for a single run', async () => {
    renderAt('/experiments/123/evaluation-runs/run-1/details', true);
    expect(await screen.findByTestId('eval-results-v4')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Published review' })).not.toBeInTheDocument();
  });

  test('keeps the single-run view when compare mode is off', async () => {
    renderAt('/experiments/123/evaluation-runs/run-1/details?compareRuns=run-2');

    expect(await screen.findByTestId('eval-results-v4')).toHaveTextContent('run:run-1');
    expect(screen.queryByTestId('eval-run-compare-view')).not.toBeInTheDocument();
  });

  test("renders the traces table inside the experiment's custom view provider", async () => {
    renderAt('/experiments/123/evaluation-runs/run-1/details');

    const provider = await screen.findByTestId('custom-view-provider');
    expect(provider).toHaveAttribute('data-experiment-id', '123');
    expect(within(provider).getByTestId('eval-results-v4')).toHaveTextContent('run:run-1');
  });

  test('keeps the artifact renderer for artifact-backed evaluation results', async () => {
    jest.mocked(useEvalRunProgress).mockReturnValue({
      ...baseProgress,
      status: MlflowRunStatus.FINISHED,
      isFinished: true,
      runTags: {
        'mlflow.loggedArtifacts': {
          key: 'mlflow.loggedArtifacts',
          value: JSON.stringify([{ type: 'table', path: '_evaluations.json' }]),
        },
      },
    });

    renderAt('/experiments/123/evaluation-runs/run-1/details');

    expect(await screen.findByTestId('eval-results-table')).toHaveTextContent('run-tags:present');
    expect(screen.queryByTestId('eval-results-v4')).not.toBeInTheDocument();
  });

  test('shows the run name, status and a link back to the list without the page shell', async () => {
    renderAt('/experiments/123/evaluation-runs/run-1/details');

    await screen.findByTestId('eval-results-v4');
    expect(screen.getByRole('heading', { name: 'my-eval-run' })).toBeInTheDocument();
    expect(screen.getByText(MlflowRunStatus.RUNNING)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Evaluation Runs' })).toHaveAttribute(
      'href',
      '/experiments/123/evaluation-runs',
    );
  });

  test('drops the in-page title and local breadcrumb in the experiment page revamp', async () => {
    renderAt('/experiments/123/evaluation-runs/run-1/details', true);

    await screen.findByTestId('eval-results-v4');
    // The run name now comes from the shared breadcrumb, not an in-page title.
    expect(screen.queryByText('my-eval-run')).not.toBeInTheDocument();
    expect(screen.queryByRole('navigation', { name: 'Breadcrumb' })).not.toBeInTheDocument();
  });

  test('drops the in-page status tag in the experiment page revamp (it moves to the breadcrumb)', async () => {
    renderAt('/experiments/123/evaluation-runs/run-1/details', true);

    // baseProgress is RUNNING / not finished: neither the title nor the status tag render in the
    // page body — both are published to the shared breadcrumb (not part of this page-level tree).
    await screen.findByTestId('eval-results-v4');
    expect(screen.queryByText(MlflowRunStatus.RUNNING)).not.toBeInTheDocument();
    expect(screen.queryByText('my-eval-run')).not.toBeInTheDocument();
  });

  test('shows the run not found page for a missing run', async () => {
    jest.mocked(useEvalRunProgress).mockReturnValue({ ...baseProgress, runName: undefined, isNotFound: true });
    renderAt('/experiments/123/evaluation-runs/missing-run/details');

    expect(await screen.findByText(/Run ID missing-run does not exist/)).toBeInTheDocument();
    expect(screen.queryByTestId('eval-results-v4')).not.toBeInTheDocument();
  });

  test("passes the experiment's baseline run and both runs' saved scores to the traces table", async () => {
    mockBaselineTag('baseline-1');
    jest.mocked(useEvalRunProgress).mockReturnValue({
      ...baseProgress,
      // The run query's copy of the experiment can be stale; the experiment query's tag wins.
      experiment: { tags: [{ key: 'mlflow.evaluation.baselineRunId', value: 'stale-1' }] } as Progress['experiment'],
      runMetrics: [{ key: 'safety/mean', value: 0.583333 }],
    });
    renderAt('/experiments/123/evaluation-runs/run-1/details');

    const table = await screen.findByTestId('eval-results-v4');
    expect(table).toHaveTextContent('baseline:baseline-1');
    expect(table).toHaveTextContent('currentSafety:0.583333');
    expect(table).toHaveTextContent('baselineSafety:0.5');
  });

  test('shows no baseline on the baseline run itself', async () => {
    mockBaselineTag('run-1');
    jest
      .mocked(useEvalRunProgress)
      .mockReturnValue({ ...baseProgress, runMetrics: [{ key: 'safety/mean', value: 0.583333 }] });
    renderAt('/experiments/123/evaluation-runs/run-1/details');

    const table = await screen.findByTestId('eval-results-v4');
    expect(table).toHaveTextContent('baseline:none');
    expect(table).toHaveTextContent('currentSafety:0.583333');
  });

  test('keys saved scores by metric name and by scorer name for the default mean', () => {
    expect(
      getEvalRunScores([
        { key: 'safety/mean', value: 0.5 },
        { key: 'relevance/mean', value: 3 },
        { key: 'relevance', value: 4 },
        { key: 'not_a_number', value: Number.NaN },
        { key: null, value: 1 },
      ]),
    ).toEqual({ 'safety/mean': 0.5, safety: 0.5, 'relevance/mean': 3, relevance: 4 });
  });

  test('shows an error instead of an empty run when the run fails to load', async () => {
    jest.mocked(useEvalRunProgress).mockReturnValue({ ...baseProgress, runName: undefined, loadError: 'Boom' });
    renderAt('/experiments/123/evaluation-runs/run-1/details');

    expect(await screen.findByText("Can't load run details")).toBeInTheDocument();
    expect(screen.getByText('Boom')).toBeInTheDocument();
    expect(screen.queryByTestId('eval-results-v4')).not.toBeInTheDocument();
  });
});
