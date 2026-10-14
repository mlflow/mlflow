import { jest, describe, beforeAll, beforeEach, test, expect } from '@jest/globals';
import { rest } from 'msw';
import { setupServer } from '../../../common/utils/setup-msw';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ExperimentEvaluationRunsPage from './ExperimentEvaluationRunsPage';
import { testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import { useLocation } from '../../../common/utils/RoutingUtils';
import { TestApolloProvider } from '../../../common/utils/TestApolloProvider';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { shouldEnableNewEvalRunsUi } from '../../../common/utils/FeatureUtils';
import { EVAL_RUNS_SORTING_QUERY_PARAM } from './hooks/useEvalRunsTableUrlState';

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableNewEvalRunsUi: jest.fn(() => false),
}));

jest.mock('../../hooks/useExperimentQuery', () => ({
  useGetExperimentQuery: jest.fn(() => ({})),
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

const SORTED_BY_NAME_DESC = JSON.stringify([{ id: 'run_name', desc: true }]);

const LocationSearch = () => <div data-testid="location-search">{useLocation().search}</div>;

describe('ExperimentEvaluationRunsPage sorting', () => {
  const server = setupServer(
    rest.post('ajax-api/2.0/mlflow/runs/search', (_req, res, ctx) =>
      res(
        ctx.json({
          runs: [createMockRun('run-b', 2), createMockRun('run-a', 1), createMockRun('run-c', 3)],
          next_page_token: null,
        }),
      ),
    ),
  );

  const renderPage = (search: string) =>
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
                        <LocationSearch />
                      </>,
                      '/experiments/:experimentId/evaluation-runs',
                    ),
                  ]}
                  initialEntries={[`/experiments/exp-1/evaluation-runs${search}`]}
                />
              </DesignSystemProvider>
            </IntlProvider>
          </MockedReduxStoreProvider>
        </QueryClientProvider>
      </TestApolloProvider>,
    );

  const getRunOrder = () => screen.getAllByText(/^run-[abc]$/).map((element) => element.textContent);
  const getSortingParam = () =>
    new URLSearchParams(screen.getByTestId('location-search').textContent ?? '').get(EVAL_RUNS_SORTING_QUERY_PARAM);

  beforeAll(() => {
    process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
    server.listen();
  });

  beforeEach(() => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
  });

  test('flag on: restores the sort from the URL', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    renderPage(`?${EVAL_RUNS_SORTING_QUERY_PARAM}=${encodeURIComponent(SORTED_BY_NAME_DESC)}`);

    await screen.findByText('run-a');
    expect(getRunOrder()).toEqual(['run-c', 'run-b', 'run-a']);
  });

  test('flag on: sorting from the header menu is kept in the URL', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    renderPage('');

    await screen.findByText('run-a');
    await userEvent.click(screen.getByRole('button', { name: 'Options for Run Name column' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Sort ascending' }));

    expect(getRunOrder()).toEqual(['run-a', 'run-b', 'run-c']);
    expect(JSON.parse(getSortingParam() ?? '')).toEqual([{ id: 'run_name', desc: false }]);
  });

  test('flag off: ignores the URL sort and keeps header sorting out of the URL', async () => {
    renderPage(`?${EVAL_RUNS_SORTING_QUERY_PARAM}=${encodeURIComponent(SORTED_BY_NAME_DESC)}`);

    await screen.findByText('run-a');
    expect(getRunOrder()).toEqual(['run-b', 'run-a', 'run-c']);

    await userEvent.click(screen.getByText('Created at'));

    expect(getRunOrder()).not.toEqual(['run-b', 'run-a', 'run-c']);
    expect(getSortingParam()).toBe(SORTED_BY_NAME_DESC);
  });
});
