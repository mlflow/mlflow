import { jest, describe, beforeAll, beforeEach, test, expect } from '@jest/globals';
import { rest } from 'msw';
import { setupServer } from '../../../common/utils/setup-msw';
import { render, screen, waitFor } from '@testing-library/react';
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

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(30000);

const createMockRun = (name: string, startTime: number) => ({
  data: { params: [], tags: [], metrics: [] },
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
    rest.post('ajax-api/2.0/mlflow/runs/search', (_req, res, ctx) =>
      res(ctx.json({ runs: [createMockRun('run-a', 2), createMockRun('run-b', 1)], next_page_token: null })),
    ),
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
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
  });

  test('flag on: clears the run selection when the experiment changes', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    await selectRunAndSwitchExperiment();

    await waitFor(async () => expect(await runACheckbox()).not.toBeChecked());
  });

  test('flag off: keeps the run selection when the experiment changes', async () => {
    await selectRunAndSwitchExperiment();

    expect(await runACheckbox()).toBeChecked();
  });
});
