import { jest, describe, beforeAll, beforeEach, test, expect } from '@jest/globals';
import type { DefaultBodyType, PathParams, ResponseResolver, RestRequest, RestContext } from 'msw';
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
import { useGetExperimentQuery } from '../../hooks/useExperimentQuery';
import { MlflowService } from '../../sdk/MlflowService';
import { MLFLOW_RUN_TYPE_TAG, MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION } from '../../constants';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { ErrorWrapper } from '@mlflow/mlflow/src/common/utils/ErrorWrapper';

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableNewEvalRunsUi: jest.fn(() => true),
}));

jest.mock('../../hooks/useExperimentQuery', () => ({
  ...jest.requireActual<typeof import('../../hooks/useExperimentQuery')>('../../hooks/useExperimentQuery'),
  useGetExperimentQuery: jest.fn(),
}));

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(30000);

interface SearchRunsRequestBody {
  filter?: string;
}

type SearchRunsHandler = ResponseResolver<RestRequest<SearchRunsRequestBody, PathParams>, RestContext, DefaultBodyType>;

const BASELINE_TAG = 'mlflow.evaluation.baselineRunId';

const createMockRun = ({ index, tags = [] }: { index: number; tags?: { key: string; value: string }[] }) => ({
  data: { params: [], tags, metrics: [{ key: 'accuracy/mean', value: index / 10, step: 0, timestamp: index }] },
  info: {
    artifact_uri: '',
    end_time: 0,
    experiment_id: 'exp-1',
    lifecycle_stage: 'active',
    run_uuid: `run-${index}`,
    run_name: `Test Run ${index}`,
    start_time: index,
    status: 'FINISHED',
  },
  inputs: { dataset_inputs: [], model_inputs: [] },
  outputs: { model_outputs: [] },
});

const mockExperimentTags = (tags: { key: string; value: string }[]) =>
  jest.mocked(useGetExperimentQuery).mockReturnValue({
    data: { experimentId: 'exp-1', tags },
    refetch: jest.fn(),
  } as unknown as ReturnType<typeof useGetExperimentQuery>);

describe('ExperimentEvaluationRunsPage baseline', () => {
  const server = setupServer();
  const requests: string[] = [];
  const useRuns = (runs: ReturnType<typeof createMockRun>[], handler?: SearchRunsHandler) =>
    server.use(
      rest.post<SearchRunsRequestBody>(
        'ajax-api/2.0/mlflow/runs/search',
        handler ?? ((_req, res, ctx) => res(ctx.json({ runs, next_page_token: null }))),
      ),
    );

  const renderPage = ({ queryClient = new QueryClient() }: { queryClient?: QueryClient } = {}) =>
    render(
      <TestApolloProvider disableCache>
        <QueryClientProvider client={queryClient}>
          <MockedReduxStoreProvider state={{ entities: { colorByRunUuid: {} } }}>
            <IntlProvider locale="en">
              <DesignSystemProvider>
                <TestRouter
                  routes={[testRoute(<ExperimentEvaluationRunsPage />, '/experiments/:experimentId/evaluation-runs')]}
                  initialEntries={['/experiments/exp-1/evaluation-runs']}
                />
              </DesignSystemProvider>
            </IntlProvider>
          </MockedReduxStoreProvider>
        </QueryClientProvider>
      </TestApolloProvider>,
    );

  // The chart legend repeats run names, so read them from the table.
  const tableRunNames = async () =>
    within(await screen.findByRole('table'))
      .getAllByText(/^Test Run \d$/)
      .map((node) => node.textContent);
  const waitForRun = (index: number) => screen.findByTestId(`eval-runs-table-cell-checkbox-run-${index}`);

  const selectRun = (index: number) =>
    userEvent.click(screen.getByTestId(`eval-runs-table-cell-checkbox-run-${index}`));

  beforeAll(() => {
    process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
    server.listen();
    server.events.on('request:start', (req) => requests.push(`${req.method} ${req.url.pathname}`));
  });

  beforeEach(() => {
    server.resetHandlers();
    requests.length = 0;
    window.localStorage.clear();
    jest.restoreAllMocks();
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    mockExperimentTags([]);
  });

  test('flag off: reads no baseline state even when a baseline tag is set', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
    mockExperimentTags([{ key: BASELINE_TAG, value: 'run-1' }]);
    useRuns([createMockRun({ index: 0 }), createMockRun({ index: 1 })]);
    const queryClient = new QueryClient();
    renderPage({ queryClient });

    await waitForRun(1);
    expect(screen.queryByText('BASELINE')).not.toBeInTheDocument();
    expect(useGetExperimentQuery).not.toHaveBeenCalled();
    expect(
      queryClient
        .getQueryCache()
        .getAll()
        .map((query) => query.queryKey),
    ).toEqual([['SEARCH_RUNS', 'exp-1', '', ['attributes.start_time DESC'], false]]);
    expect(requests.filter((request) => /users\/|runs\/get/.test(request))).toEqual([]);
  });

  test('pins the baseline with a BASELINE pill', async () => {
    mockExperimentTags([{ key: BASELINE_TAG, value: 'run-0' }]);
    useRuns([createMockRun({ index: 2 }), createMockRun({ index: 1 }), createMockRun({ index: 0 })]);
    renderPage();

    expect(await screen.findByText('BASELINE')).toBeInTheDocument();
    expect(await tableRunNames()).toEqual(['Test Run 0', 'Test Run 2', 'Test Run 1']);
  });

  test('keeps a baseline that the filters exclude, fetched on its own', async () => {
    mockExperimentTags([{ key: BASELINE_TAG, value: 'run-9' }]);
    useRuns([createMockRun({ index: 0 })]);
    server.use(
      rest.get('ajax-api/2.0/mlflow/runs/get', (_req, res, ctx) => res(ctx.json({ run: createMockRun({ index: 9 }) }))),
    );
    renderPage();

    expect(await screen.findByText('BASELINE')).toBeInTheDocument();
    expect(await tableRunNames()).toEqual(['Test Run 9', 'Test Run 0']);
  });

  test('shows the baseline action only when exactly one run is selected', async () => {
    useRuns([createMockRun({ index: 0 }), createMockRun({ index: 1 })]);
    const setExperimentTag = jest.spyOn(MlflowService, 'setExperimentTag').mockResolvedValue({});
    renderPage();
    await waitForRun(0);

    expect(screen.queryByRole('button', { name: 'Set baseline' })).not.toBeInTheDocument();
    await selectRun(0);
    const setBaselineButton = await screen.findByRole('button', { name: 'Set baseline' });
    await waitFor(() => expect(setBaselineButton).toBeEnabled());
    expect(setBaselineButton).not.toHaveTextContent('Set baseline');
    await userEvent.hover(setBaselineButton);
    expect(await screen.findAllByText('Set baseline')).not.toHaveLength(0);

    await selectRun(1);
    expect(screen.queryByRole('button', { name: 'Set baseline' })).not.toBeInTheDocument();
    await selectRun(1);

    await userEvent.click(screen.getByRole('button', { name: 'Set baseline' }));
    await waitFor(() =>
      expect(setExperimentTag).toHaveBeenCalledWith({ experiment_id: 'exp-1', key: BASELINE_TAG, value: 'run-0' }),
    );
    expect(screen.getByTestId('eval-runs-table-cell-checkbox-run-0')).not.toBeChecked();
  });

  test('disables setting a baseline when filtering drops the retained selection', async () => {
    useRuns([], (req, res, ctx) =>
      res(
        ctx.json({
          runs: req.body.filter?.includes('filtered-out') ? [] : [createMockRun({ index: 0 })],
          next_page_token: null,
        }),
      ),
    );
    renderPage();
    await waitForRun(0);
    await selectRun(0);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Set baseline' })).toBeEnabled());

    await userEvent.click(screen.getByRole('combobox', { name: 'Search evaluation runs' }));
    await userEvent.click(await screen.findByRole('button', { name: 'Add a Filter' }));
    await userEvent.click(await screen.findByRole('combobox', { name: 'Filter field' }));
    await userEvent.click(screen.getByRole('option', { name: 'Run name' }));
    // The string value input is a typeahead textbox in the OSS design system.
    await userEvent.click(screen.getByRole('textbox', { name: 'Filter value' }));
    await userEvent.paste('filtered-out');
    await userEvent.click(screen.getByRole('button', { name: 'Apply' }));

    await waitFor(() => expect(screen.queryByTestId('eval-runs-table-cell-checkbox-run-0')).not.toBeInTheDocument());
    expect(screen.getByRole('button', { name: 'Set baseline' })).toBeDisabled();
  });

  test('offers the baseline action for a run the server returned without tags', async () => {
    // The OSS server omits empty repeated fields, so a run with no tags comes back as `data: {}`.
    useRuns([{ ...createMockRun({ index: 0 }), data: {} } as unknown as ReturnType<typeof createMockRun>]);
    renderPage();
    await waitForRun(0);

    await selectRun(0);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Set baseline' })).toBeEnabled());
  });

  test('disables setting an issue detection run as baseline', async () => {
    useRuns([
      createMockRun({ index: 0, tags: [{ key: MLFLOW_RUN_TYPE_TAG, value: MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION }] }),
    ]);
    renderPage();
    await waitForRun(0);

    await selectRun(0);
    const setBaseline = await screen.findByRole('button', { name: 'Set baseline' });
    expect(setBaseline).toBeDisabled();
    await userEvent.hover(setBaseline);
    expect(await screen.findAllByText('Issue detection runs cannot be used as a baseline')).not.toHaveLength(0);
  });

  test('replaces or clears an existing baseline', async () => {
    mockExperimentTags([{ key: BASELINE_TAG, value: 'run-1' }]);
    useRuns([createMockRun({ index: 0 }), createMockRun({ index: 1 })]);
    const setExperimentTag = jest.spyOn(MlflowService, 'setExperimentTag').mockResolvedValue({});
    const deleteExperimentTag = jest.spyOn(MlflowService, 'deleteExperimentTag').mockResolvedValue(undefined);
    renderPage();
    await screen.findByText('BASELINE');

    await selectRun(0);
    const replaceButton = await screen.findByRole('button', { name: 'Replace baseline' });
    await waitFor(() => expect(replaceButton).toBeEnabled());
    await userEvent.click(replaceButton);
    await waitFor(() =>
      expect(setExperimentTag).toHaveBeenCalledWith({ experiment_id: 'exp-1', key: BASELINE_TAG, value: 'run-0' }),
    );
    expect(screen.getByTestId('eval-runs-table-cell-checkbox-run-0')).not.toBeChecked();

    await selectRun(1);
    await userEvent.click(await screen.findByRole('button', { name: 'Clear baseline' }));
    await waitFor(() =>
      expect(deleteExperimentTag).toHaveBeenCalledWith({ experiment_id: 'exp-1', key: BASELINE_TAG }),
    );
  });

  test("shows the server's message when it rejects the baseline write", async () => {
    useRuns([createMockRun({ index: 0 })]);
    // Basic auth answers a READ user's write with a plain-text 403.
    const setExperimentTag = jest
      .spyOn(MlflowService, 'setExperimentTag')
      .mockRejectedValue(new ErrorWrapper('Permission denied', 403));
    const notify = jest.spyOn(Utils, 'logErrorAndNotifyUser').mockImplementation(() => {});
    renderPage();
    await waitForRun(0);

    await selectRun(0);
    await userEvent.click(await screen.findByRole('button', { name: 'Set baseline' }));

    await waitFor(() => expect(notify).toHaveBeenCalled());
    expect((notify.mock.calls[0][0] as ErrorWrapper).renderHttpError()).toBe('Permission denied');
    expect(setExperimentTag).toHaveBeenCalled();
    expect(screen.queryByText('BASELINE')).not.toBeInTheDocument();
  });
});
