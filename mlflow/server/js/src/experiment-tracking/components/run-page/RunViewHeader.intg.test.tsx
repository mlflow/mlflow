import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { within } from '../../../common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';
import { MlflowService } from '../../sdk/MlflowService';
import { IntlProvider } from 'react-intl';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { DesignSystemProvider } from '@databricks/design-system';
import { RunViewHeader } from './RunViewHeader';
import { ExperimentKind, ExperimentPageTabName } from '@mlflow/mlflow/src/experiment-tracking/constants';
import { EXPERIMENT_KIND_TAG_KEY } from '../../utils/ExperimentKindUtils';
import { TestRouter, testRoute, waitForRoutesToBeRendered } from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';
import Routes from '../../routes';
import { prefixRouteWithWorkspace } from '@mlflow/mlflow/src/workspaces/utils/WorkspaceUtils';

jest.mock('../../sdk/MlflowService', () => ({
  MlflowService: {
    searchRuns: jest.fn(),
  },
}));

jest.mock('../../../common/utils/FeatureUtils', () => ({
  shouldEnableExperimentPageHeaderV2: () => true,
  shouldUseRenamedUnifiedTracesTab: () => false,
  shouldDisableReproduceRunButton: () => false,
  shouldEnableWorkflowBasedNavigation: () => false,
  shouldEnableImprovedEvalRunsComparison: () => false,
}));

describe('RunViewHeader - integration test', () => {
  const testRunUuid = 'test-run-uuid';
  const testExperimentId = 'test-experiment-id';
  const testRunName = 'Test Run';

  beforeEach(() => {
    jest.clearAllMocks();
  });

  const renderTestComponent = (props: any) => {
    const defaultProps = {
      runDisplayName: testRunName,
      runUuid: testRunUuid,
      runTags: {},
      runParams: {},
      experiment: {
        experimentId: testExperimentId,
        name: 'Test Experiment',
        tags: [],
      },
      handleRenameRunClick: jest.fn(),
      handleDeleteRunClick: jest.fn(),
      registeredModelVersionSummaries: [],
      ...props,
    };

    const TestComponent = () => {
      return (
        <TestRouter
          routes={[
            testRoute(
              <DesignSystemProvider>
                <QueryClientProvider
                  client={
                    new QueryClient({
                      logger: {
                        error: () => {},
                        log: () => {},
                        warn: () => {},
                      },
                    })
                  }
                >
                  <RunViewHeader {...defaultProps} />
                </QueryClientProvider>
              </DesignSystemProvider>,
            ),
          ]}
        />
      );
    };

    return render(
      <IntlProvider locale="en">
        <TestComponent />
      </IntlProvider>,
    );
  };

  it('routes to evaluation-runs tab when experiment is GenAI and has no model outputs', async () => {
    const experimentWithGenAITag = {
      experimentId: testExperimentId,
      name: 'Test Experiment',
      tags: [{ key: EXPERIMENT_KIND_TAG_KEY, value: ExperimentKind.GENAI_DEVELOPMENT }],
    };

    const runOutputs = {
      modelOutputs: [],
    };

    renderTestComponent({
      experiment: experimentWithGenAITag,
      runOutputs,
    });

    await waitForRoutesToBeRendered();

    const experimentLink = screen.getByTestId('experiment-observatory-link-runs');
    expect(experimentLink.textContent).toBe('Evaluations');
    const expectedPath = prefixRouteWithWorkspace(
      Routes.getExperimentPageTabRoute(testExperimentId, ExperimentPageTabName.EvaluationRuns),
    );
    expect(experimentLink.getAttribute('href')).toBe(expectedPath);
  });

  it('routes to runs tab when experiment is GenAI but has model outputs', async () => {
    const experimentWithGenAITag = {
      experimentId: testExperimentId,
      name: 'Test Experiment',
      tags: [{ key: EXPERIMENT_KIND_TAG_KEY, value: ExperimentKind.GENAI_DEVELOPMENT }],
    };

    const runOutputs = {
      modelOutputs: [{ someOutput: 'value' }],
    };

    renderTestComponent({
      experiment: experimentWithGenAITag,
      runOutputs,
    });

    await waitForRoutesToBeRendered();

    const experimentLink = screen.getByTestId('experiment-observatory-link-runs');
    expect(experimentLink.textContent).toBe('Runs');
    const expectedPath = prefixRouteWithWorkspace(
      Routes.getExperimentPageTabRoute(testExperimentId, ExperimentPageTabName.Runs),
    );
    expect(experimentLink.getAttribute('href')).toBe(expectedPath);
  });

  it('routes to runs tab when experiment is not GenAI', async () => {
    const experimentWithoutGenAITag = {
      experimentId: testExperimentId,
      name: 'Test Experiment',
      tags: [],
    };

    renderTestComponent({
      experiment: experimentWithoutGenAITag,
    });

    await waitForRoutesToBeRendered();

    const experimentLink = screen.getByTestId('experiment-observatory-link-runs');
    expect(experimentLink.textContent).toBe('Runs');
    const expectedPath = prefixRouteWithWorkspace(
      Routes.getExperimentPageTabRoute(testExperimentId, ExperimentPageTabName.Runs),
    );
    expect(experimentLink.getAttribute('href')).toBe(expectedPath);
  });

  it('renders the run switcher trigger button when activeTab is provided', async () => {
    renderTestComponent({ activeTab: 'model-metrics' });

    await waitForRoutesToBeRendered();

    expect(screen.getByRole('button', { name: 'Switch run' })).toBeInTheDocument();
  });

  it('does not render the run switcher trigger button when activeTab is not provided', async () => {
    renderTestComponent({});

    await waitForRoutesToBeRendered();

    expect(screen.queryByRole('button', { name: 'Switch run' })).not.toBeInTheDocument();
  });

  it('hides compare buttons when supportsComparison is false even when onCompareRun is provided', async () => {
    jest.mocked(MlflowService.searchRuns).mockResolvedValueOnce({
      runs: [
        {
          info: {
            runUuid: 'run-2',
            runName: 'Run Two',
            experimentId: testExperimentId,
            artifactUri: '',
            endTime: 0,
            lifecycleStage: 'active',
            startTime: 0,
            status: 'FINISHED',
          },
          data: { metrics: [], params: [], tags: [] },
        },
      ],
    } as any);

    renderTestComponent({
      activeTab: 'overview',
      supportsComparison: false,
      onCompareRun: jest.fn(),
    });

    await waitForRoutesToBeRendered();
    await userEvent.click(screen.getByRole('button', { name: 'Switch run' }));
    const run2Item = await screen.findByRole('menuitemcheckbox', { name: /Run Two/ });
    expect(within(run2Item).queryByRole('button')).not.toBeInTheDocument();
  });
});
