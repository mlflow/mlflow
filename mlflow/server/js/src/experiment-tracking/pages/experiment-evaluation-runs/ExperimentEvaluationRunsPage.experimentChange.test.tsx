import { jest, describe, beforeAll, beforeEach, test, expect } from '@jest/globals';
import { rest } from 'msw';
import { setupServer } from '../../../common/utils/setup-msw';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ExperimentEvaluationRunsPage from './ExperimentEvaluationRunsPage';
import { testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import { useNavigate } from '../../../common/utils/RoutingUtils';
import { TestApolloProvider } from '../../../common/utils/TestApolloProvider';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { shouldEnableNewEvalRunsUi } from '../../../common/utils/FeatureUtils';

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableNewEvalRunsUi: jest.fn(() => false),
}));

// Records which runs the page publishes as selected, and for which experiment, on every render.
const mockPublishedSelections: { experimentId?: string; runUuids: string[] }[] = [];
jest.mock('@mlflow/mlflow/src/assistant', () => ({
  ...jest.requireActual<typeof import('@mlflow/mlflow/src/assistant')>('@mlflow/mlflow/src/assistant'),
  useRegisterSelectedIds: (_key: string, selection: Record<string, boolean>) => {
    const { useParams: mockUseParams } = jest.requireActual<typeof import('../../../common/utils/RoutingUtils')>(
      '../../../common/utils/RoutingUtils',
    );
    // eslint-disable-next-line react-hooks/rules-of-hooks -- runs inside the page's render
    const { experimentId } = mockUseParams();
    mockPublishedSelections.push({
      experimentId,
      runUuids: Object.keys(selection).filter((runUuid) => selection[runUuid]),
    });
  },
}));

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(30000);

const createMockRun = (name: string, startTime: number, metrics: string[] = []) => ({
  data: { params: [], tags: [], metrics: metrics.map((key) => ({ key, value: 0.5, step: 0, timestamp: startTime })) },
  info: {
    artifact_uri: '',
    end_time: 0,
    experiment_id: 'exp-1',
    lifecycle_stage: '',
    run_uuid: name,
    run_name: name,
    start_time: startTime,
    status: 'FINISHED',
  },
  inputs: { dataset_inputs: [], model_inputs: [] },
  outputs: { model_outputs: [] },
});

const SwitchExperimentButton = () => {
  const navigate = useNavigate();
  return <button onClick={() => navigate('/experiments/exp-2/evaluation-runs')}>Switch experiment</button>;
};

describe('ExperimentEvaluationRunsPage experiment change', () => {
  const server = setupServer(
    rest.post('ajax-api/2.0/mlflow/runs/search', (req, res, ctx) => {
      const experimentId = (req.body as { experiment_ids?: string[] }).experiment_ids?.[0];
      return res(
        ctx.json({
          runs:
            experimentId === 'exp-2'
              ? [createMockRun('run-a', 3, ['shared']), createMockRun('run-c', 2, ['shared'])]
              : [createMockRun('run-a', 2, ['shared', 'only-in-exp-1']), createMockRun('run-b', 1, ['shared'])],
          next_page_token: null,
        }),
      );
    }),
  );

  const renderPage = () =>
    render(
      <TestApolloProvider disableCache>
        <QueryClientProvider client={new QueryClient()}>
          <MockedReduxStoreProvider state={{ entities: { colorByRunUuid: {} } }}>
            <IntlProvider locale="en">
              <DesignSystemProvider>
                <TestRouter
                  routes={[
                    testRoute(
                      <>
                        <ExperimentEvaluationRunsPage />
                        <SwitchExperimentButton />
                      </>,
                      '/experiments/:experimentId/evaluation-runs',
                    ),
                  ]}
                  initialEntries={['/experiments/exp-1/evaluation-runs']}
                />
              </DesignSystemProvider>
            </IntlProvider>
          </MockedReduxStoreProvider>
        </QueryClientProvider>
      </TestApolloProvider>,
    );

  const runACheckbox = () => screen.findByTestId('eval-runs-table-cell-checkbox-run-a');

  const selectRunAndSwitchExperiment = async () => {
    renderPage();
    await userEvent.click(await runACheckbox());
    expect(await runACheckbox()).toBeChecked();
    await userEvent.click(screen.getByRole('button', { name: 'Switch experiment' }));
  };

  beforeAll(() => {
    process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
    server.listen();
  });

  beforeEach(() => {
    mockPublishedSelections.length = 0;
    window.localStorage.clear();
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
  });

  test('flag on: clears the run selection when the experiment changes', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    await selectRunAndSwitchExperiment();

    await screen.findByTestId('eval-runs-table-cell-checkbox-run-c');
    expect(screen.queryByRole('button', { name: /^Actions/ })).not.toBeInTheDocument();
  });

  test('flag on: never publishes the previous experiment selection for the new experiment', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    await selectRunAndSwitchExperiment();
    await screen.findByTestId('eval-runs-table-cell-checkbox-run-c');

    expect(
      mockPublishedSelections.some(({ experimentId, runUuids }) => experimentId === 'exp-1' && runUuids.length),
    ).toBe(true);
    expect(
      mockPublishedSelections.filter(({ experimentId, runUuids }) => experimentId === 'exp-2' && runUuids.length),
    ).toEqual([]);
  });

  test('flag on: column choices start fresh in the new experiment', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    renderPage();
    await userEvent.hover(await screen.findByRole('columnheader', { name: /^shared/ }));
    await userEvent.click(screen.getByRole('button', { name: 'Options for shared column' }));
    await userEvent.click(screen.getByRole('menuitem', { name: 'Hide column' }));
    expect(screen.queryByRole('columnheader', { name: /^shared/ })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Switch experiment' }));
    await screen.findByTestId('eval-runs-table-cell-checkbox-run-c');

    expect(screen.getByRole('columnheader', { name: /^shared/ })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Display' }));
    await userEvent.click(screen.getByRole('menuitem', { name: 'Columns' }));
    expect(screen.queryByRole('menuitemcheckbox', { name: 'only-in-exp-1' })).not.toBeInTheDocument();
  });

  test('flag off: keeps the run selection when the experiment changes', async () => {
    await selectRunAndSwitchExperiment();

    expect(await runACheckbox()).toBeChecked();
  });
});
