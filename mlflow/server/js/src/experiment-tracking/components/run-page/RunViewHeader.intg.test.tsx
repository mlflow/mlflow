import { render, screen } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { DesignSystemProvider } from '@databricks/design-system';
import { RunViewHeader } from './RunViewHeader';
import { ExperimentKind, ExperimentPageTabName } from '@mlflow/mlflow/src/experiment-tracking/constants';
import { EXPERIMENT_KIND_TAG_KEY } from '../../utils/ExperimentKindUtils';
import { TestRouter, testRoute, waitForRoutesToBeRendered } from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';
import Routes from '../../routes';

jest.mock('../../../common/utils/FeatureUtils', () => ({
  shouldEnableExperimentPageHeaderV2: () => true,
  shouldUseRenamedUnifiedTracesTab: () => false,
  shouldDisableReproduceRunButton: () => false,
}));

describe('RunViewHeader - integration test', () => {
  const testRunUuid = 'test-run-uuid';
  const testExperimentId = 'test-experiment-id';
  const testRunName = 'Test Run';

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
    const expectedPath = Routes.getExperimentPageTabRoute(testExperimentId, ExperimentPageTabName.EvaluationRuns);
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
    const expectedPath = Routes.getExperimentPageTabRoute(testExperimentId, ExperimentPageTabName.Runs);
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
    const expectedPath = Routes.getExperimentPageTabRoute(testExperimentId, ExperimentPageTabName.Runs);
    expect(experimentLink.getAttribute('href')).toBe(expectedPath);
  });
});
