import { jest, describe, beforeAll, beforeEach, test, expect } from '@jest/globals';
import { rest } from 'msw';
import { setupServer } from '../../../common/utils/setup-msw';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ExperimentEvaluationRunsPage from './ExperimentEvaluationRunsPage';
import { testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import { TestApolloProvider } from '../../../common/utils/TestApolloProvider';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { shouldEnableNewEvalRunsUi } from '../../../common/utils/FeatureUtils';
import { useLocation } from '../../../common/utils/RoutingUtils';

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableNewEvalRunsUi: jest.fn(() => false),
}));

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(30000);

const createMockRun = (name: string, startTime: number, accuracy: number) => ({
  data: {
    params: [],
    tags: [],
    metrics: [{ key: 'accuracy/mean', value: accuracy, timestamp: startTime, step: 0 }],
  },
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

const CompareTarget = () => {
  const location = useLocation();
  return <div data-testid="compare-target">{`${location.pathname}${location.search}`}</div>;
};

describe('ExperimentEvaluationRunsPage new eval runs UI', () => {
  const server = setupServer(
    rest.post('ajax-api/2.0/mlflow/runs/search', (_req, res, ctx) =>
      res(
        ctx.json({
          runs: [createMockRun('run-a', 2, 0.8), createMockRun('run-b', 1, 0.6)],
          next_page_token: null,
        }),
      ),
    ),
  );

  const renderPage = ({
    queryClient = new QueryClient(),
    initialEntry = '/experiments/exp-1/evaluation-runs',
  }: { queryClient?: QueryClient; initialEntry?: string } = {}) =>
    render(
      <TestApolloProvider disableCache>
        <QueryClientProvider client={queryClient}>
          <MockedReduxStoreProvider state={{ entities: { colorByRunUuid: {} } }}>
            <IntlProvider locale="en">
              <DesignSystemProvider>
                <TestRouter
                  routes={[
                    testRoute(<ExperimentEvaluationRunsPage />, '/experiments/:experimentId/evaluation-runs'),
                    testRoute(<CompareTarget />, '/experiments/:experimentId/evaluation-runs/:runUuid/details'),
                  ]}
                  initialEntries={[initialEntry]}
                />
              </DesignSystemProvider>
            </IntlProvider>
          </MockedReduxStoreProvider>
        </QueryClientProvider>
      </TestApolloProvider>,
    );

  const openActionsForRunA = async (actionsName: string | RegExp) => {
    renderPage();
    await userEvent.click(await screen.findByTestId('eval-runs-table-cell-checkbox-run-a'));
    await userEvent.click(await screen.findByRole('button', { name: actionsName }));
    return screen.findByRole('menu');
  };

  beforeAll(() => {
    process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
    server.listen();
  });

  beforeEach(() => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
  });

  test('flag on: shows the single-row toolbar with the date range control', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    renderPage();

    expect(await screen.findByRole('button', { name: 'Date' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Refresh evaluation runs' })).toBeInTheDocument();
  });

  test('flag off: keeps the previous toolbar', async () => {
    renderPage();

    expect(await screen.findByTestId('eval-runs-table-cell-checkbox-run-a')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Date' })).not.toBeInTheDocument();
  });

  const queryKeysAfterLoad = async (queryClient: QueryClient) => {
    await screen.findByTestId('eval-runs-table-cell-checkbox-run-a');
    return queryClient
      .getQueryCache()
      .getAll()
      .map((query) => query.queryKey);
  };

  test('flag off: an active search creates no unfiltered-count query', async () => {
    const queryClient = new QueryClient();
    // The URL search param is leak bait: the new UI would treat it as an active filter.
    renderPage({ queryClient, initialEntry: '/experiments/exp-1/evaluation-runs?evalRunsSearchFilter=run' });

    expect(await queryKeysAfterLoad(queryClient)).toEqual([
      ['SEARCH_RUNS', 'exp-1', '', ['attributes.start_time DESC'], false, true],
    ]);
  });

  test('flag on: an active search adds the unfiltered-count query', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    const queryClient = new QueryClient();
    renderPage({ queryClient, initialEntry: '/experiments/exp-1/evaluation-runs?evalRunsSearchFilter=run' });

    await waitFor(async () =>
      expect(await queryKeysAfterLoad(queryClient)).toContainEqual(['EVAL_RUNS_UNFILTERED_COUNT', 'exp-1']),
    );
  });

  test('flag on: shows the recent runs chart above the table', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    renderPage();

    expect(await screen.findByText('Earliest run')).toBeInTheDocument();
    expect(screen.getByText('Latest run')).toBeInTheDocument();
  });

  test('flag off: does not show the recent runs chart', async () => {
    renderPage();

    expect(await screen.findByTestId('eval-runs-table-cell-checkbox-run-a')).toBeInTheDocument();
    expect(screen.queryByText('Earliest run')).not.toBeInTheDocument();
  });

  test('flag on: the actions menu filters the chart and Compare moves to its own button', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    renderPage();
    await userEvent.click(await screen.findByTestId('eval-runs-table-cell-checkbox-run-a'));

    expect(screen.getByRole('button', { name: 'Compare (1)' })).toBeDisabled();
    const actions = screen.getByRole('button', { name: 'Actions' });
    expect(actions).not.toHaveTextContent('Actions');
    await userEvent.click(actions);
    const menu = await screen.findByRole('menu');

    expect(within(menu).getByText('Filter chart to selected runs')).toBeInTheDocument();
    expect(within(menu).getByText('Delete runs')).toBeInTheDocument();
    expect(within(menu).queryByRole('menuitem', { name: /Compare/ })).not.toBeInTheDocument();
  });

  test('flag on: Compare with two runs selected opens the comparison page', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    renderPage();
    await userEvent.click(await screen.findByTestId('eval-runs-table-cell-checkbox-run-a'));
    await userEvent.click(screen.getByTestId('eval-runs-table-cell-checkbox-run-b'));

    await userEvent.click(screen.getByRole('button', { name: 'Compare (2)' }));

    expect(await screen.findByTestId('compare-target')).toHaveTextContent(
      '/experiments/exp-1/evaluation-runs/run-a/details?compare=true&compareFrom=runs&compareRuns=run-b',
    );
  });

  test('flag off: the actions menu keeps Compare', async () => {
    const menu = await openActionsForRunA(/^Actions$/);

    expect(within(menu).getByText('Compare')).toBeInTheDocument();
    expect(within(menu).queryByText('Filter chart to selected runs')).not.toBeInTheDocument();
  });
});
