import { afterEach, describe, expect, jest, test } from '@jest/globals';
import { fireEvent, render, screen, within } from '@databricks/web-shared/test-utils/render';
import userEvent from '@testing-library/user-event';
import { QueryClient } from '@databricks/web-shared/query-client';
import type { MockedResponse } from '../../../common/utils/graphQLHooks';
import type { GetRun } from '../../../graphql/__generated__/graphql';
import { GET_RUN_QUERY } from '../../components/run-page/hooks/useGetRunQuery';
import {
  EvalRunComparisonConfiguration,
  type EvalRunComparisonConfigurationProps,
} from './EvalRunComparisonConfiguration';
import { aggregateEvalRunConfigurationTraces } from './evalRunConfigurationData';
import { createComparisonRun, createComparisonTrace } from './evalRunComparisonTestUtils';
import { EvalRunComparisonTestProviders } from './EvalRunComparisonTestProviders';
import type { EvalRunConfigurationTraceData } from './useEvalRunComparisonData';

const runMock = (runUuid: string): MockedResponse<GetRun> => ({
  request: { query: GET_RUN_QUERY, variables: { data: { runId: runUuid } } },
  result: {
    data: {
      mlflowGetRun: {
        __typename: 'MlflowGetRunResponse',
        apiError: null,
        run: createComparisonRun(runUuid),
      },
    },
  },
});

const traceRun = (runUuid: string, tokens = 100, cost = 0.3): EvalRunConfigurationTraceData => {
  const trace = createComparisonTrace(runUuid, { quality: false });
  trace.trace_metadata = {
    'mlflow.trace.tokenUsage': JSON.stringify({ total_tokens: tokens }),
    'mlflow.trace.cost': JSON.stringify({ total_cost: cost }),
  };
  return {
    runUuid,
    runName: `Run ${runUuid}`,
    stats: aggregateEvalRunConfigurationTraces([trace], false),
    status: 'ready',
  };
};

const clients: QueryClient[] = [];
afterEach(() => clients.splice(0).forEach((client) => client.clear()));

const renderConfiguration = (
  props: Partial<EvalRunComparisonConfigurationProps> = {},
  mocks = ['baseline', 'candidate', 'third'].map(runMock),
) => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  clients.push(client);
  const defaults: EvalRunComparisonConfigurationProps = {
    experimentId: 'experiment-id',
    runs: [traceRun('baseline'), traceRun('candidate', 60, 0.2)],
    noComparison: false,
    onRetryTraces: jest.fn<() => Promise<void>>().mockResolvedValue(),
  };
  const view = (current: Partial<EvalRunComparisonConfigurationProps>) => (
    <EvalRunComparisonTestProviders client={client} apolloMocks={mocks}>
      <EvalRunComparisonConfiguration {...defaults} {...props} {...current} />
    </EvalRunComparisonTestProviders>
  );
  const result = render(view({}));
  return {
    ...result,
    change: (current: Partial<EvalRunComparisonConfigurationProps>) => result.rerender(view(current)),
  };
};

const rowCells = (name: string) =>
  within(screen.getByRole('row', { name: new RegExp(`^${name}`) }))
    .getAllByRole('cell')
    .slice(1);

describe('configuration comparison section', () => {
  test('loads metadata separately and renders relative changes and tied minima', async () => {
    renderConfiguration();
    expect(screen.getByRole('heading', { name: 'Run information' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'Run' })).toBeInTheDocument();
    expect(screen.getAllByText('Loading configuration').length).toBeGreaterThan(0);
    expect(await screen.findByText('dataset-baseline')).toBeInTheDocument();
    expect(screen.getByText('dataset-candidate')).toBeInTheDocument();
    expect(rowCells('Date created')[0]).toHaveTextContent(/ago/);
    expect(rowCells('Total tokens')[1]).toHaveTextContent('60Lowest value-40%');
    expect(rowCells('Total cost')[1]).toHaveTextContent('≈$0.2000Lowest value-33.33%');
    expect(rowCells('Errors').every((cell) => within(cell).queryByText('Lowest value'))).toBe(true);
    expect(screen.queryByText('Model version')).not.toBeInTheDocument();
  });

  test('updates columns and baseline deltas as runs are added, removed, swapped, and reordered', async () => {
    const view = renderConfiguration();
    expect(await screen.findByText('dataset-baseline')).toBeInTheDocument();
    view.change({
      runs: [traceRun('baseline'), traceRun('candidate', 60), traceRun('third', 80)],
    });
    expect(await screen.findByText('dataset-third')).toBeInTheDocument();
    expect(screen.getAllByRole('columnheader')).toHaveLength(4);
    expect(rowCells('Total tokens')[2]).toHaveTextContent('80-20%');
    view.change({ runs: [traceRun('third', 80), traceRun('baseline')] });
    expect(screen.getAllByRole('columnheader')).toHaveLength(3);
    expect(
      within(screen.getByRole('columnheader', { name: /Run third Baseline$/ })).getByText('Baseline'),
    ).toBeInTheDocument();
    expect(rowCells('Total tokens')[1]).toHaveTextContent('100+25%');
    expect(screen.queryByText('dataset-candidate')).not.toBeInTheDocument();
  });

  test('links each run name to its single-run detail page', async () => {
    renderConfiguration();
    expect(await screen.findByText('dataset-baseline')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Run baseline' })).toHaveAttribute(
      'href',
      '/experiments/experiment-id/evaluation-runs/baseline/details',
    );
    expect(screen.getByRole('link', { name: 'Run candidate' })).toHaveAttribute(
      'href',
      '/experiments/experiment-id/evaluation-runs/candidate/details',
    );
  });

  test('cycles configuration sorting from high-to-low to low-to-high to original order', async () => {
    renderConfiguration({
      runs: [traceRun('baseline', 100), traceRun('candidate', 120), traceRun('third', 60)],
    });
    expect(await screen.findByText('dataset-third')).toBeInTheDocument();
    const sortButton = screen.getByRole('button', {
      name: 'Sort Total tokens High → low',
    });
    expect(sortButton).toHaveAttribute('aria-pressed', 'false');
    await userEvent.click(sortButton);
    expect(screen.getByText('High → low')).toBeInTheDocument();
    expect(sortButton).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getAllByRole('link', { name: /^Run / }).map((link) => link.textContent)).toEqual([
      'Run baseline',
      'Run candidate',
      'Run third',
    ]);
    await userEvent.click(screen.getByRole('button', { name: 'Sort Total tokens Low → high' }));
    expect(screen.getByText('Low → high')).toBeInTheDocument();
    expect(screen.getAllByRole('link', { name: /^Run / }).map((link) => link.textContent)).toEqual([
      'Run baseline',
      'Run third',
      'Run candidate',
    ]);
    await userEvent.click(screen.getByRole('button', { name: 'Sort Total tokens Original order' }));
    expect(screen.getAllByRole('link', { name: /^Run / }).map((link) => link.textContent)).toEqual([
      'Run baseline',
      'Run candidate',
      'Run third',
    ]);
  });

  test('uses date and text direction labels for metadata sorting', async () => {
    renderConfiguration();
    expect(await screen.findByText('dataset-baseline')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Sort Date created Newest first' }));
    expect(screen.getByText('Newest first')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Sort Dataset Z → A' }));
    expect(screen.getByText('Z → A')).toBeInTheDocument();
  });

  test('navigates to trace view with the metric outcome filter when a delta is clicked', async () => {
    const onNavigateToTraceView = jest.fn();
    renderConfiguration({ onNavigateToTraceView });
    expect(await screen.findByText('dataset-baseline')).toBeInTheDocument();
    const improvement = within(rowCells('Total tokens')[1]).getByRole('button', { name: '-40%' });
    expect(improvement).toHaveAttribute('tabindex', '0');
    await userEvent.hover(improvement);
    expect((await screen.findAllByText(/Click here to filter to improvements in this run/)).length).toBeGreaterThan(0);
    await userEvent.click(improvement);
    expect(onNavigateToTraceView).toHaveBeenCalledWith({
      columnId: 'tokens',
      runUuid: 'candidate',
      outcome: 'improvement',
    });
  });

  test('keeps deltas informational when outcome filtering is unavailable', () => {
    renderConfiguration({ onNavigateToTraceView: jest.fn(), enableOutcomeFilter: false });
    expect(within(rowCells('Total tokens')[1]).getByText('-40%')).toBeInTheDocument();
    expect(within(rowCells('Total tokens')[1]).queryByRole('button', { name: '-40%' })).not.toBeInTheDocument();
    expect(within(rowCells('Total tokens')[1]).getByRole('status', { name: '-40%' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Compare traces row by row' })).toBeInTheDocument();
  });

  test('explains that a regression bubble filters the run', async () => {
    renderConfiguration({
      runs: [traceRun('baseline', 100), traceRun('candidate', 120)],
      onNavigateToTraceView: jest.fn(),
    });
    const regression = within(rowCells('Total tokens')[1]).getByRole('button', { name: '+20%' });
    await userEvent.hover(regression);
    expect((await screen.findAllByText(/Click here to filter to regressions in this run/)).length).toBeGreaterThan(0);
  });

  test('labels partial usage and withholds differences and minimum highlights', async () => {
    const candidate = traceRun('candidate', 60);
    if (!candidate.stats) throw new Error('Missing fixture statistics');
    candidate.stats = {
      ...candidate.stats,
      traceCount: 2,
      tracesWithTokens: 1,
    };
    renderConfiguration({ runs: [traceRun('baseline'), candidate] });
    const tokens = rowCells('Total tokens');
    expect(tokens[1]).toHaveTextContent('60Partial');
    expect(tokens[1]).not.toHaveTextContent('-40%');
    expect(within(tokens[0]).queryByText('Lowest value')).not.toBeInTheDocument();
    within(tokens[1]).getByLabelText('Usage coverage').focus();
    expect(await screen.findByRole('tooltip')).toHaveTextContent('Total tokens available for 1 of 2 loaded traces.');
  });

  test('withholds all rankings and affected deltas when a run may be truncated', () => {
    const candidate = traceRun('candidate', 60);
    if (!candidate.stats) throw new Error('Missing fixture statistics');
    candidate.stats.isTruncated = true;
    renderConfiguration({ runs: [traceRun('baseline'), candidate] });
    expect(screen.getByText(/At least one run reached the trace search limit/)).toBeInTheDocument();
    expect(screen.queryByText('Lowest value')).not.toBeInTheDocument();
    expect(rowCells('Total tokens')[1]).toHaveTextContent(/^60$/);
  });

  test.each(['noComparison'] as const)('retains metadata while traces are unavailable: %s', async (reason) => {
    renderConfiguration({
      [reason]: true,
      runs: [
        {
          runUuid: 'baseline',
          runName: 'Run baseline',
          status: 'unavailable',
        },
      ],
    });
    expect(await screen.findByText('dataset-baseline')).toBeInTheDocument();
    expect(rowCells('Traces')[0]).toHaveTextContent('—');
    expect(rowCells('Errors')[0]).toHaveTextContent('—');
    expect(screen.getByText('Add a comparison run to load trace statistics.')).toBeInTheDocument();
  });

  test('keeps fetched statistics during metadata failure and retries only the failed configuration', async () => {
    const candidate = runMock('candidate');
    renderConfiguration({}, [
      runMock('baseline'),
      { request: candidate.request, error: new Error('Unavailable') },
      candidate,
    ]);
    expect(await screen.findByText('Configuration unavailable')).toBeInTheDocument();
    expect(rowCells('Dataset')[1]).toHaveTextContent('—');
    expect(rowCells('Total tokens')[1]).toHaveTextContent('60Lowest value-40%');
    await userEvent.setup().click(
      screen.getByRole('button', {
        name: 'Retry configuration for Run candidate',
      }),
    );
    expect(await screen.findByText('dataset-candidate')).toBeInTheDocument();
    expect(screen.queryByText('Configuration unavailable')).not.toBeInTheDocument();
  });

  test('renders empty trace results as zero counts but unavailable usage', () => {
    renderConfiguration({
      runs: [
        {
          runUuid: 'baseline',
          runName: 'Run baseline',
          status: 'ready',
          stats: aggregateEvalRunConfigurationTraces([], false),
        },
      ],
    });
    expect(rowCells('Traces')[0]).toHaveTextContent(/^0$/);
    expect(rowCells('Errors')[0]).toHaveTextContent(/^0$/);
    expect(rowCells('Total tokens')[0]).toHaveTextContent('—');
    expect(rowCells('Total cost')[0]).toHaveTextContent('—');
  });

  test('does not fabricate usage totals when loaded traces have no usage metadata', () => {
    renderConfiguration({
      runs: [
        {
          runUuid: 'baseline',
          runName: 'Run baseline',
          status: 'ready',
          stats: aggregateEvalRunConfigurationTraces([createComparisonTrace('baseline', {})], false),
        },
      ],
    });
    expect(rowCells('Traces')[0]).toHaveTextContent(/^1$/);
    expect(rowCells('Total tokens')[0]).toHaveTextContent(/^—$/);
    expect(rowCells('Total cost')[0]).toHaveTextContent(/^—$/);
  });

  test.each(['hover', 'focus'])('explains the absolute difference on %s', async (interaction) => {
    renderConfiguration();
    expect(await screen.findByText('dataset-baseline')).toBeInTheDocument();
    const delta = within(rowCells('Total tokens')[1]).getByText('-40%');
    if (interaction === 'hover') {
      await userEvent.hover(delta);
    } else {
      fireEvent.focus(delta);
    }
    expect(await screen.findByRole('tooltip')).toHaveTextContent('Absolute difference: -40');
  });

  test('rounds total cost and its absolute difference to four decimal places', async () => {
    renderConfiguration({
      runs: [traceRun('baseline', 100, 0.123456), traceRun('candidate', 60, 0.234567)],
    });
    expect(await screen.findByText('dataset-baseline')).toBeInTheDocument();
    const costCell = rowCells('Total cost')[1];
    expect(costCell).toHaveTextContent('$0.2346');
    fireEvent.focus(within(costCell).getByText('+90%'));
    expect(await screen.findByRole('tooltip')).toHaveTextContent(/^Absolute difference: [+][$]0.1111$/);
  });

  test('shows N/A with the absolute difference when the baseline is zero', async () => {
    renderConfiguration({
      runs: [traceRun('baseline', 0, 0), traceRun('candidate', 10, 0.1)],
    });
    expect(await screen.findByText('dataset-baseline')).toBeInTheDocument();
    const costCell = rowCells('Total cost')[1];
    const delta = within(costCell).getByText('N/A');
    fireEvent.focus(delta);
    expect(await screen.findByRole('tooltip')).toHaveTextContent(
      'Relative change is unavailable because the baseline is zero. Absolute difference: +$0.1000',
    );
    expect(costCell).toHaveTextContent('$0.1000N/A');
  });

  test('labels the total cost as an estimate', async () => {
    renderConfiguration();
    expect(await screen.findByText('dataset-baseline')).toBeInTheDocument();
    const infoButton = screen.getByLabelText('This is an estimate');
    await userEvent.hover(infoButton);
    expect(await screen.findByRole('tooltip')).toHaveTextContent(/^This is an estimate$/);
  });

  test('hovering the cost value shows the estimation explanation', async () => {
    renderConfiguration();
    expect(await screen.findByText('dataset-baseline')).toBeInTheDocument();
    // Hover the ≈-prefixed cost value (first run's ≈$0.3000 baseline cell).
    const costValues = screen.getAllByText(/^≈\$/);
    await userEvent.hover(costValues[0]);
    expect(await screen.findByRole('tooltip')).toHaveTextContent(
      'Cost is estimated from token usage and public model pricing.',
    );
  });
});
