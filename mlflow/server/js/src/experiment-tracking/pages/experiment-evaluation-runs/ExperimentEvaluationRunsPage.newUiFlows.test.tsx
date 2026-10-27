import { jest, describe, beforeAll, beforeEach, test, expect } from '@jest/globals';
import type { DefaultBodyType, PathParams, ResponseResolver, RestRequest, RestContext } from 'msw';
import { rest } from 'msw';
import { setupServer } from '../../../common/utils/setup-msw';
import { render, screen, waitFor, fireEvent, act, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ExperimentEvaluationRunsPage from './ExperimentEvaluationRunsPage';
import { testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import { useLocation, useNavigate } from '../../../common/utils/RoutingUtils';
import { TestApolloProvider } from '../../../common/utils/TestApolloProvider';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { shouldEnableNewEvalRunsUi } from '../../../common/utils/FeatureUtils';

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableNewEvalRunsUi: jest.fn(() => true),
}));

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(30000);

interface SearchRunsRequestBody {
  experiment_ids?: string[];
  filter?: string;
  max_results?: number;
  page_token?: string;
}

type SearchRunsHandler = ResponseResolver<RestRequest<SearchRunsRequestBody, PathParams>, RestContext, DefaultBodyType>;

const createMockRun = ({
  index,
  datasetName,
  metrics = [],
  startTime = index,
}: {
  index: number;
  datasetName?: string;
  metrics?: [string, number][];
  startTime?: number;
}) => ({
  data: {
    params: [],
    tags: [],
    metrics: metrics.map(([key, value]) => ({ key, value, step: 0, timestamp: startTime })),
  },
  info: {
    artifact_uri: '',
    end_time: 0,
    experiment_id: 'exp-1',
    lifecycle_stage: '',
    run_uuid: `run-${index}`,
    run_name: `Test Run ${index}`,
    start_time: startTime,
    status: 'FINISHED',
  },
  inputs: {
    dataset_inputs: datasetName ? [{ dataset: { digest: `digest-${index}`, name: datasetName }, tags: [] }] : [],
    model_inputs: [],
  },
  outputs: { model_outputs: [] },
});

const NavigationControls = () => {
  const navigate = useNavigate();
  const location = useLocation();
  return (
    <>
      <button onClick={() => navigate('/experiments/exp-2/evaluation-runs')}>Go to experiment 2</button>
      <button onClick={() => navigate('/experiments/exp-1/evaluation-runs')}>Go to experiment 1</button>
      <div data-testid="location-search">{location.search}</div>
    </>
  );
};

describe('ExperimentEvaluationRunsPage new eval runs UI flows', () => {
  const server = setupServer();
  const useSearchHandler = (handler: SearchRunsHandler) =>
    server.use(rest.post<SearchRunsRequestBody>('ajax-api/2.0/mlflow/runs/search', handler));

  const renderTestComponent = ({ initialEntry = '/experiments/exp-1/evaluation-runs' } = {}) =>
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
                        <NavigationControls />
                      </>,
                      '/experiments/:experimentId/evaluation-runs',
                    ),
                  ]}
                  initialEntries={[initialEntry]}
                />
              </DesignSystemProvider>
            </IntlProvider>
          </MockedReduxStoreProvider>
        </QueryClientProvider>
      </TestApolloProvider>,
    );

  const prepareTable = async () => {
    const table = await screen.findByRole('table', {}, { timeout: 10000 });
    Object.defineProperty(table, 'scrollHeight', { configurable: true, value: 1000 });
    Object.defineProperty(table, 'scrollTop', { configurable: true, writable: true, value: 0 });
    return table;
  };

  const selectFilterOption = async (comboboxName: string, optionText: string) => {
    await userEvent.click(screen.getByRole('combobox', { name: comboboxName }));
    await userEvent.click(screen.getByRole('option', { name: optionText }));
  };

  const openUnifiedFilterEditor = async () => {
    await userEvent.click(screen.getByRole('combobox', { name: 'Search evaluation runs' }));
    await userEvent.click(await screen.findByRole('button', { name: 'Add a Filter' }));
    await screen.findByRole('combobox', { name: 'Filter field' });
  };

  beforeAll(() => {
    process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
    server.listen();
  });

  beforeEach(() => {
    server.resetHandlers();
    window.localStorage.clear();
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
  });

  test('keeps chart assessment selections separate by experiment', async () => {
    useSearchHandler((req, res, ctx) =>
      res(
        ctx.json({
          runs: [
            createMockRun({
              index: 0,
              metrics: [[req.body.experiment_ids?.[0] === 'exp-2' ? 'destination' : 'source', 0.5]],
            }),
          ],
        }),
      ),
    );
    renderTestComponent();
    await prepareTable();
    await screen.findByTestId('line-source');
    await userEvent.click(screen.getByRole('button', { name: 'All assessments' }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'source' }));
    expect(screen.queryByTestId('line-source')).not.toBeInTheDocument();
    await userEvent.keyboard('{Escape}');

    await userEvent.click(screen.getByRole('button', { name: 'Go to experiment 2' }));
    await screen.findByTestId('line-destination');
    expect(screen.queryByTestId('line-source')).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Go to experiment 1' }));
    await screen.findByRole('button', { name: 'No assessments' });
    expect(screen.queryByTestId('line-source')).not.toBeInTheDocument();
  });

  test('restores selected chart assessments after a page remount', async () => {
    useSearchHandler((_req, res, ctx) =>
      res(
        ctx.json({
          runs: [
            createMockRun({
              index: 0,
              metrics: [
                ['quality', 0.8],
                ['judge', 0.9],
              ],
            }),
          ],
        }),
      ),
    );

    const { unmount } = renderTestComponent();
    await screen.findByTestId('line-judge');
    await userEvent.click(screen.getByRole('button', { name: 'All assessments' }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'quality' }));
    expect(screen.getByRole('button', { name: 'judge' })).toBeInTheDocument();
    unmount();

    renderTestComponent();
    await screen.findByTestId('line-judge');
    expect(screen.queryByTestId('line-quality')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'judge' })).toBeInTheDocument();
  });

  test('explains an empty time range and resets it to all time', async () => {
    useSearchHandler((req, res, ctx) =>
      res(
        ctx.json({
          runs: req.body.filter?.includes('attributes.start_time') ? [] : [createMockRun({ index: 0 })],
          next_page_token: null,
        }),
      ),
    );

    renderTestComponent();
    await screen.findByText('Test Run 0');

    await userEvent.click(screen.getByRole('button', { name: 'Date' }));
    await userEvent.click(screen.getByRole('option', { name: 'Last hour' }));

    expect(await screen.findByText('No evaluation runs in this time range')).toBeInTheDocument();
    expect(screen.queryByAltText('No runs found')).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Show all time' }));

    expect(await screen.findByText('Test Run 0')).toBeInTheDocument();
    expect(screen.queryByText('No evaluation runs in this time range')).not.toBeInTheDocument();
  });

  test('moves a relative time range forward to now on refresh', async () => {
    const hour = 60 * 60 * 1000;
    const cutoffs: number[] = [];
    useSearchHandler((req, res, ctx) => {
      const cutoff = /attributes\.start_time > (\d+)/.exec(req.body.filter ?? '')?.[1];
      if (cutoff) {
        cutoffs.push(Number(cutoff));
      }
      return res(ctx.json({ runs: [createMockRun({ index: 0 })], next_page_token: null }));
    });
    const nowSpy = jest.spyOn(Date, 'now').mockReturnValue(10 * hour);

    renderTestComponent({ initialEntry: '/experiments/exp-1/evaluation-runs?startTimeLabel=LAST_HOUR' });
    await screen.findByText('Test Run 0');
    expect(cutoffs).toEqual([9 * hour]);

    nowSpy.mockReturnValue(11 * hour);
    await userEvent.click(screen.getByRole('button', { name: 'Refresh evaluation runs' }));

    await waitFor(() => expect(cutoffs).toEqual([9 * hour, 10 * hour]));
    nowSpy.mockRestore();
  });

  test('waits for the unfiltered runs before choosing an empty state', async () => {
    let releaseUnfilteredRuns: () => void = () => {};
    const unfilteredRunsReleased = new Promise<void>((resolve) => {
      releaseUnfilteredRuns = resolve;
    });
    let filteredRequestCount = 0;
    useSearchHandler(async (req, res, ctx) => {
      if (req.body.filter?.includes('attributes.start_time')) {
        filteredRequestCount++;
        return res(ctx.json({ runs: [], next_page_token: null }));
      }
      await unfilteredRunsReleased;
      return res(ctx.json({ runs: [createMockRun({ index: 0 })], next_page_token: null }));
    });

    // A saved time range filters the very first load, so both queries start together.
    renderTestComponent({ initialEntry: '/experiments/exp-1/evaluation-runs?startTimeLabel=LAST_HOUR' });
    await waitFor(() => expect(filteredRequestCount).toBeGreaterThan(0));
    await act(async () => {});

    expect(screen.queryByAltText('No runs found')).not.toBeInTheDocument();
    expect(screen.queryByText('No evaluation runs in this time range')).not.toBeInTheDocument();

    releaseUnfilteredRuns();
    expect(await screen.findByText('No evaluation runs in this time range')).toBeInTheDocument();
    expect(screen.queryByAltText('No runs found')).not.toBeInTheDocument();
  });

  test('clears the search, filters, and time range together', async () => {
    useSearchHandler((req, res, ctx) =>
      res(ctx.json({ runs: req.body.filter ? [] : [createMockRun({ index: 0 })], next_page_token: null })),
    );

    renderTestComponent({
      initialEntry:
        '/experiments/exp-1/evaluation-runs?evalRunsSearchFilter=metrics.accuracy%20%3E%200.9&startTimeLabel=LAST_HOUR',
    });
    await userEvent.click(await screen.findByRole('button', { name: 'Clear filters' }));

    expect(await screen.findByText('Test Run 0')).toBeInTheDocument();
    const params = new URLSearchParams(screen.getByTestId('location-search').textContent ?? '');
    expect(params.get('evalRunsSearchFilter')).toBeNull();
    expect(params.get('startTimeLabel')).toBe('ALL');
  });

  test('keeps chart and table selections independent through pagination and filtering', async () => {
    const assessments = (count: number): [string, number][] =>
      Array.from({ length: count }, (_, i) => [`assessment-${i + 1}`, (i + 1) / 10]);
    const firstPageRuns = [createMockRun({ index: 0, datasetName: 'alpha', metrics: assessments(6) })];
    const secondPageRuns = [createMockRun({ index: 1, datasetName: 'beta', metrics: assessments(7) })];
    useSearchHandler((req, res, ctx) => {
      if ((req.body.filter ?? '').includes("dataset.name = 'alpha'")) {
        return res(ctx.json({ runs: firstPageRuns, next_page_token: null }));
      }
      return res(
        ctx.json({
          runs: req.body.page_token ? secondPageRuns : firstPageRuns,
          next_page_token: req.body.page_token ? null : 'next-page',
        }),
      );
    });

    renderTestComponent();
    const table = await prepareTable();
    await within(table).findByText('Test Run 0');

    await userEvent.click(screen.getByRole('radio', { name: 'Bar chart' }));
    expect(screen.getAllByRole('img').map((bar) => bar.getAttribute('aria-label'))).toEqual([
      'Test Run 0, assessment-1: 10%',
      'Test Run 0, assessment-2: 20%',
      'Test Run 0, assessment-3: 30%',
      'Test Run 0, assessment-4: 40%',
      'Test Run 0, assessment-5: 50%',
    ]);

    // The chart defaults to five assessments while the table shows every metric.
    expect(screen.getByRole('button', { name: '5 assessments' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'assessment-1' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'assessment-5' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'assessment-6' })).toBeInTheDocument();

    // Chart changes leave table columns unchanged.
    await userEvent.click(screen.getByRole('button', { name: '5 assessments' }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'assessment-6' }));
    expect(screen.getByRole('button', { name: 'All assessments' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'assessment-6' })).toBeInTheDocument();
    expect(screen.getByRole('img', { name: 'Test Run 0, assessment-6: 60%' })).toBeInTheDocument();

    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'assessment-6' }));
    expect(screen.queryByRole('img', { name: 'Test Run 0, assessment-6: 60%' })).not.toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'assessment-6' })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'assessment-6' }));

    // Table changes leave the chart selection unchanged.
    await userEvent.click(screen.getByRole('button', { name: 'Display' }));
    await userEvent.click(screen.getByRole('menuitem', { name: 'Columns' }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'assessment-1' }));
    expect(screen.getByRole('button', { name: 'All assessments' })).toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'assessment-1' })).not.toBeInTheDocument();
    expect(screen.getByRole('img', { name: 'Test Run 0, assessment-1: 10%' })).toBeInTheDocument();

    fireEvent.scroll(table, { target: { scrollTop: 1000 } });
    await screen.findByText('Test Run 1');

    // Pagination introduces assessment-7 without resetting the user's explicit choices.
    expect(screen.queryByRole('columnheader', { name: 'assessment-1' })).not.toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'assessment-6' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'assessment-7' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '6 assessments' })).toBeInTheDocument();
    expect(screen.getByRole('img', { name: 'Test Run 1, assessment-6: 60%' })).toBeInTheDocument();
    expect(screen.queryByRole('img', { name: 'Test Run 1, assessment-7: 70%' })).not.toBeInTheDocument();

    // Filtering the data back to the first run also keeps the explicit metric choices intact.
    await userEvent.keyboard('{Escape}');
    await userEvent.keyboard('{Escape}');
    await openUnifiedFilterEditor();
    await selectFilterOption('Filter field', 'Dataset');
    // The string value input is a typeahead textbox in the OSS design system.
    await userEvent.click(screen.getByRole('textbox', { name: 'Filter value' }));
    await userEvent.paste('alpha');
    await userEvent.click(screen.getByRole('option', { name: 'alpha' }));
    await userEvent.click(screen.getByRole('button', { name: 'Apply' }));
    await waitFor(() => expect(screen.queryByText('Test Run 1')).not.toBeInTheDocument());
    expect(screen.queryByRole('columnheader', { name: 'assessment-1' })).not.toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'assessment-6' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'All assessments' })).toBeInTheDocument();
    expect(screen.getByTestId('bar-chart')).toHaveAttribute('data-count', '6');
    expect(screen.getByRole('img', { name: 'Test Run 0, assessment-6: 60%' })).toBeInTheDocument();
    expect(screen.queryByRole('img', { name: 'Test Run 1, assessment-6: 60%' })).not.toBeInTheDocument();
    // The longest flow in this file; give it room on slower CI machines.
  }, 120000);

  test('selects a table run from a bar and filters the chart only on request', async () => {
    useSearchHandler((_req, res, ctx) =>
      res(
        ctx.json({
          runs: [
            createMockRun({ index: 0, metrics: [['assessment-1', 0.9]] }),
            createMockRun({ index: 1, metrics: [['assessment-1', 0.8]] }),
          ],
          next_page_token: null,
        }),
      ),
    );

    renderTestComponent();
    await prepareTable();
    await userEvent.click(screen.getByRole('radio', { name: 'Bar chart' }));
    await userEvent.click(screen.getByRole('img', { name: 'Test Run 0, assessment-1: 90%' }));

    expect(screen.getByRole('checkbox', { name: 'Select run Test Run 0' })).toBeChecked();
    expect(screen.getByRole('img', { name: 'Test Run 1, assessment-1: 80%' })).toBeInTheDocument();

    await userEvent.click(screen.getByRole('img', { name: 'Test Run 0, assessment-1: 90%' }));
    expect(screen.getByRole('checkbox', { name: 'Select run Test Run 0' })).not.toBeChecked();
    await userEvent.click(screen.getByRole('img', { name: 'Test Run 0, assessment-1: 90%' }));
    expect(screen.getByRole('checkbox', { name: 'Select run Test Run 0' })).toBeChecked();

    await userEvent.click(screen.getByRole('button', { name: 'Actions' }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'Filter chart to selected runs' }));
    expect(screen.getByRole('img', { name: 'Test Run 0, assessment-1: 90%' })).toBeInTheDocument();
    expect(screen.queryByRole('img', { name: 'Test Run 1, assessment-1: 80%' })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('checkbox', { name: 'Select run Test Run 0' }));
    expect(screen.getByRole('img', { name: 'Test Run 1, assessment-1: 80%' })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('checkbox', { name: 'Select run Test Run 1' }));
    await userEvent.click(screen.getByRole('button', { name: 'Actions' }));
    expect(screen.getByRole('menuitemcheckbox', { name: 'Filter chart to selected runs' })).not.toBeChecked();
  });

  test('does not explain a failed filtered search as an empty result', async () => {
    let filteredRequestCount = 0;
    let unfilteredRequestCount = 0;
    useSearchHandler((req, res, ctx) => {
      if (req.body.filter?.includes('attributes.start_time')) {
        filteredRequestCount++;
        return res(ctx.status(500), ctx.json({ error_code: 'INTERNAL_ERROR', message: 'Search failed' }));
      }
      unfilteredRequestCount++;
      return res(ctx.json({ runs: [createMockRun({ index: 0 })], next_page_token: null }));
    });

    renderTestComponent({ initialEntry: '/experiments/exp-1/evaluation-runs?startTimeLabel=LAST_HOUR' });
    // Wait for both searches so no request is still in flight when the next test starts.
    await waitFor(() => {
      expect(filteredRequestCount).toBeGreaterThan(0);
      expect(unfilteredRequestCount).toBeGreaterThan(0);
    });
    await act(async () => {});

    expect(screen.queryByText('No evaluation runs in this time range')).not.toBeInTheDocument();
    expect(screen.queryByAltText('No runs found')).not.toBeInTheDocument();
  });
});
