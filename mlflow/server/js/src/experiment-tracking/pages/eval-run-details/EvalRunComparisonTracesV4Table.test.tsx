import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { configure, fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useEffect } from 'react';
import { rest } from 'msw';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { setupServer } from '../../../common/utils/setup-msw';
import { setupTestRouter } from '../../../common/utils/RoutingTestUtils';
import { useLocation } from '../../../common/utils/RoutingUtils';
import { renderTracesV4Page } from '../../components/experiment-page/components/traces-v4/test-utils/renderTracesV4Page';
import {
  makeFeedbackAssessment,
  makeTrace,
} from '../../components/experiment-page/components/traces-v4/test-utils/mockTraces';
import { EvalRunComparisonTracesV4Table } from './EvalRunComparisonTracesV4Table';

const EXPERIMENT_ID = 'exp-1';
const PATH = '/ml/experiments/:experimentId/evaluation-runs/:runUuid/details';
const COMPARE_URL = `/ml/experiments/${EXPERIMENT_ID}/evaluation-runs/baseline/details?compare=true&compareRuns=candidate`;
// OSS traces live in the experiment, so their URL id is the bare trace id.
const makeRunTrace = (id: string, runUuid: string, over: Partial<ModelTraceInfoV3> = {}) =>
  makeTrace(id, {
    trace_location: { type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: EXPERIMENT_ID } },
    trace_metadata: { 'mlflow.sourceRun': runUuid },
    ...over,
  });
const scored = (id: string, runUuid: string, input: string, quality: number, over: Partial<ModelTraceInfoV3> = {}) =>
  makeRunTrace(id, runUuid, {
    request_preview: input,
    response_preview: `${runUuid} answer to ${input}`,
    assessments: [
      makeFeedbackAssessment('quality', quality, { metadata: { 'mlflow.assessment.sourceRunId': runUuid } }),
    ],
    ...over,
  });

// Real trace ids, so a selected trace passes the exact-id check before it opens.
const traceId = (name: string) => `tr-${name.padEnd(32, '0')}`;

const state: {
  tracesByRun: Record<string, ModelTraceInfoV3[]>;
  failRun?: string;
  inputsReady: Promise<void>;
  inputsRequested: boolean;
} = { tracesByRun: {}, inputsReady: Promise.resolve(), inputsRequested: false };
let lastSearch = '';

const server = setupServer(
  rest.post('/ajax-api/3.0/mlflow/traces/search', async (req, res, ctx) => {
    const body = (await req.json()) as { filter?: string };
    const runUuid = body.filter?.match(/attributes\.run_id = '([^']*)'/)?.[1] ?? '';
    if (runUuid === state.failRun) {
      return res(ctx.status(500), ctx.json({ message: 'Unavailable' }));
    }
    return res(ctx.json({ traces: state.tracesByRun[runUuid] ?? [], next_page_token: undefined }));
  }),
  rest.get('/ajax-api/3.0/mlflow/traces/batchGet', async (_req, res, ctx) => {
    state.inputsRequested = true;
    await state.inputsReady;
    return res(ctx.json({ traces: [] }));
  }),
  rest.post('/ajax-api/2.0/mlflow/runs/search', (_req, res, ctx) =>
    res(
      ctx.json({
        runs: [
          { info: { run_uuid: 'baseline', run_name: 'Baseline run' } },
          { info: { run_uuid: 'candidate', run_name: 'Candidate run' } },
        ],
      }),
    ),
  ),
  rest.post('/ajax-api/3.0/mlflow/datasets/search', (_req, res, ctx) => res(ctx.json({ datasets: [] }))),
);

// The first render of the table, its toolbar and its drawer providers is slow under jsdom.
configure({ asyncUtilTimeout: 15000 });
const findMatchedRows = () => screen.findByText('2 comparison rows');

const LocationSpy = () => {
  const search = useLocation().search;
  useEffect(() => {
    lastSearch = search;
  }, [search]);
  return null;
};

describe('EvalRunComparisonTracesV4Table', () => {
  const { history } = setupTestRouter();

  const renderTable = (initialUrl = COMPARE_URL) =>
    renderTracesV4Page({
      initialUrl,
      routes: [
        {
          path: PATH,
          element: (
            <>
              <LocationSpy />
              <EvalRunComparisonTracesV4Table experimentId={EXPERIMENT_ID} runUuids={['baseline', 'candidate']} />
            </>
          ),
        },
      ],
      history,
      experimentId: EXPERIMENT_ID,
    });

  beforeAll(() => {
    process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
    server.listen();
  });

  beforeEach(() => {
    window.localStorage.setItem('mlflow.detectIssues.guidanceShown_v1', 'true');
    state.tracesByRun = {
      baseline: [
        scored(traceId('b1'), 'baseline', 'First question', 0.8),
        scored(traceId('b2'), 'baseline', 'Second question', 0.4),
      ],
      candidate: [
        scored(traceId('c1'), 'candidate', 'First question', 0.6),
        scored(traceId('c2'), 'candidate', 'Second question', 0.9),
      ],
    };
    state.failRun = undefined;
    state.inputsReady = Promise.resolve();
    state.inputsRequested = false;
    lastSearch = '';
  });

  afterEach(() => {
    jest.clearAllMocks();
    window.localStorage.clear();
  });

  afterAll(() => {
    delete process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'];
    server.close();
  });

  test('shows one row per input with a sub-row per run and counts the matched rows', async () => {
    renderTable();
    await findMatchedRows();

    expect(await screen.findByRole('cell', { name: 'First question' })).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: 'Second question' })).toBeInTheDocument();
    const rowGroups = screen.getAllByRole('rowgroup');
    expect(rowGroups).toHaveLength(2);
    for (const group of rowGroups) {
      expect(within(group).getByRole('row', { name: /Baseline run/ })).toBeInTheDocument();
      expect(within(group).getByRole('row', { name: /Candidate run/ })).toBeInTheDocument();
    }
    expect(screen.getByText('2 comparison rows')).toBeInTheDocument();
    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
  });

  test('summarizes each scorer per run in its header, with the change from the baseline', async () => {
    renderTable();
    await findMatchedRows();

    const qualityHeader = await screen.findByRole('columnheader', { name: /quality/ });
    await waitFor(() => expect(within(qualityHeader).getByRole('group', { name: 'Baseline run' })).toBeVisible());
    expect(within(qualityHeader).getByRole('group', { name: 'Baseline run' })).toHaveTextContent('0.6');
    expect(within(qualityHeader).getByRole('group', { name: 'Candidate run' })).toHaveTextContent('0.75');
    expect(within(qualityHeader).getByRole('group', { name: 'Candidate run' })).toHaveTextContent('+0.15');
  });

  test('switches a numeric scorer between Mean, Median, Min and Max from its column menu', async () => {
    renderTable();
    await findMatchedRows();

    const qualityHeader = await screen.findByRole('columnheader', { name: /quality/ });
    await userEvent.click(within(qualityHeader).getByRole('button', { name: /Column options/ }));
    expect(screen.getByRole('menuitem', { name: 'Mean' })).toBeVisible();
    await userEvent.click(screen.getByRole('menuitem', { name: 'Max' }));

    await waitFor(() =>
      expect(within(qualityHeader).getByRole('group', { name: 'Baseline run' })).toHaveTextContent('MAX'),
    );
    expect(within(qualityHeader).getByRole('group', { name: 'Baseline run' })).toHaveTextContent('0.8');
    expect(within(qualityHeader).getByRole('group', { name: 'Candidate run' })).toHaveTextContent('0.9');
  });

  test('opens a run’s trace in the trace drawer from its sub-row', async () => {
    renderTable();
    await findMatchedRows();

    // Run names arrive with the experiment's runs, after the traces.
    const candidateRow = await waitFor(() => {
      const row = screen
        .getAllByRole('row', { name: /Candidate run/ })
        .find((candidate) => within(candidate).queryAllByRole('columnheader').length === 0);
      if (!row) {
        throw new Error('Expected a Candidate run sub-row');
      }
      return row;
    });
    await userEvent.click(within(candidateRow).getByRole('link', { name: 'Candidate run' }));

    await waitFor(() => expect(new URLSearchParams(lastSearch).get('selectedEvaluationId')).toBe(traceId('c1')));
    expect(new URLSearchParams(lastSearch).get('compareRuns')).toBe('candidate');
    // Opening the trace drawer over the full comparison table can pass 20s on loaded CI runners; a
    // per-test timeout avoids the lint-forbidden global jest.setTimeout.
  }, 60000);

  test('narrows the rows to the search and offers to clear it when nothing matches', async () => {
    renderTable(`${COMPARE_URL}&comparisonQ=nothing%20matches`);

    expect(await screen.findByText(/No traces match/)).toBeInTheDocument();
  });

  test('explains a comparison that cannot load every run', async () => {
    state.failRun = 'candidate';
    renderTable();

    expect(
      await screen.findByText('Could not load traces for all selected runs. Retry to load the comparison.'),
    ).toBeInTheDocument();
  });

  test('describes compared runs without traces', async () => {
    state.tracesByRun = {};
    renderTable();

    expect(await screen.findByText('No traces in this evaluation run')).toBeInTheDocument();
  });
  const outputsInOrder = () =>
    screen.getAllByRole('rowgroup').flatMap((group) =>
      within(group)
        .queryAllByText(/ answer to /)
        .map((cell) => cell.textContent),
    );

  const useDurationTraces = () => {
    state.tracesByRun = {
      baseline: [
        scored(traceId('b1'), 'baseline', 'Fast question', 0.8, { execution_duration: '1s', state: 'ERROR' }),
        scored(traceId('b2'), 'baseline', 'Slow question', 0.4, { execution_duration: '9s' }),
      ],
      candidate: [
        scored(traceId('c1'), 'candidate', 'Fast question', 0.6, { execution_duration: '2s' }),
        scored(traceId('c2'), 'candidate', 'Slow question', 0.9, { execution_duration: '3s' }),
      ],
    };
  };

  test('sorts whole rows by the baseline value from a column header', async () => {
    useDurationTraces();
    renderTable(`${COMPARE_URL}&comparisonSort=duration&comparisonDir=asc`);
    await findMatchedRows();
    await waitFor(() => expect(screen.getAllByRole('rowgroup')).toHaveLength(2));
    expect(outputsInOrder()).toEqual([
      'baseline answer to Fast question',
      'candidate answer to Fast question',
      'baseline answer to Slow question',
      'candidate answer to Slow question',
    ]);

    const header = screen.getByRole('columnheader', { name: /^Duration/ });
    await userEvent.hover(within(header).getByRole('button', { name: 'Sort' }));
    expect(await screen.findByRole('tooltip')).toHaveTextContent(
      'Groups are sorted by the baseline trace’s value. Ties use the other runs in displayed order. Missing values appear last.',
    );
    await userEvent.click(within(header).getByRole('button', { name: 'Column options' }));
    await userEvent.click(screen.getByRole('menuitem', { name: 'Sort descending' }));

    expect(new URLSearchParams(lastSearch).get('comparisonSort')).toBe('duration');
    expect(new URLSearchParams(lastSearch).get('comparisonDir')).toBe('desc');
    expect(new URLSearchParams(lastSearch).get('compareRuns')).toBe('candidate');
    expect(outputsInOrder()).toEqual([
      'baseline answer to Slow question',
      'candidate answer to Slow question',
      'baseline answer to Fast question',
      'candidate answer to Fast question',
    ]);
  });

  test('keeps a whole row when one trace matches a saved filter, and recomputes its header summary', async () => {
    useDurationTraces();
    const filters = JSON.stringify([{ field: 'state', operator: '=', value: 'ERROR' }]);
    renderTable(`${COMPARE_URL}&comparisonFilters=${encodeURIComponent(filters)}`);

    await waitFor(() =>
      expect(outputsInOrder()).toEqual(['baseline answer to Fast question', 'candidate answer to Fast question']),
    );
    expect(screen.getByText('1 comparison row')).toBeInTheDocument();
    const qualityHeader = screen.getByRole('columnheader', { name: /quality/ });
    expect(within(qualityHeader).getByRole('group', { name: 'Baseline run' })).toHaveTextContent('0.8');
    expect(within(qualityHeader).getByRole('group', { name: 'Candidate run' })).toHaveTextContent('0.6');
  });

  test('offers one Filters button with only the fields the comparison can match', async () => {
    renderTable();
    expect(await findMatchedRows()).toBeVisible();
    const filterButtons = screen.getAllByRole('button', { name: /Filters/ });
    expect(filterButtons).toHaveLength(1);
    await userEvent.click(filterButtons[0]);
    await userEvent.click(await screen.findByRole('combobox', { name: /Filter field/ }));
    expect(await screen.findByRole('option', { name: 'Duration' })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Span name' })).not.toBeInTheDocument();
  });

  test('explains comparison filtering and keeps an incomplete filter in the editor', async () => {
    renderTable();
    expect(await findMatchedRows()).toBeVisible();
    await userEvent.click(screen.getByRole('button', { name: /Filters/ }));
    expect(screen.getByText('Keep a comparison row when one trace matches all filters and the search.')).toBeVisible();
    await userEvent.click(screen.getByRole('button', { name: 'Apply filters' }));

    expect(screen.getByText('Complete or remove unsupported filters before applying.')).toBeVisible();
    expect(screen.getByRole('button', { name: 'Apply filters' })).toBeVisible();
    expect(new URLSearchParams(lastSearch).get('comparisonFilters')).toBeNull();
  });

  test('offers recovery for an unsupported saved filter', async () => {
    const filters = JSON.stringify([{ field: 'span_name', operator: '=', value: 'ignored' }]);
    renderTable(`${COMPARE_URL}&comparisonFilters=${encodeURIComponent(filters)}`);

    expect(await screen.findByText('Complete or remove unsupported filters before applying.')).toBeVisible();
    expect(await screen.findByText('No traces match your filters')).toBeVisible();
    await userEvent.click(screen.getByRole('button', { name: 'Clear filters' }));

    expect(await findMatchedRows()).toBeVisible();
    expect(new URLSearchParams(lastSearch).get('comparisonFilters')).toBeNull();
  });

  test('clears both search and filters from the no-results state', async () => {
    const filters = JSON.stringify([{ field: 'state', operator: '=', value: 'ERROR' }]);
    renderTable(`${COMPARE_URL}&comparisonQ=absent&comparisonFilters=${encodeURIComponent(filters)}`);

    expect(await screen.findByText('No traces match your filters')).toBeVisible();
    await userEvent.click(screen.getByText('Clear filters'));

    expect(await findMatchedRows()).toBeVisible();
    expect(new URLSearchParams(lastSearch).get('comparisonFilters')).toBeNull();
    expect(new URLSearchParams(lastSearch).get('comparisonQ')).toBeNull();
    expect(screen.getByRole('textbox', { name: 'Search traces by id or input' })).toHaveValue('');
  });

  test('pairs repeated inputs across runs, newest first', async () => {
    const repeated = (runUuid: string, index: number, minute: number) =>
      scored(traceId(`${runUuid === 'baseline' ? 'b' : 'c'}${index}`), runUuid, 'Same question', 0.5, {
        response_preview: `${runUuid} answer ${index}`,
        request_time: new Date(Date.UTC(2026, 0, 1, 0, minute)).toISOString(),
      });
    state.tracesByRun = {
      baseline: [repeated('baseline', 1, 1), repeated('baseline', 2, 2)],
      candidate: [repeated('candidate', 1, 1), repeated('candidate', 2, 2)],
    };
    renderTable();
    await findMatchedRows();
    expect(
      screen.getAllByRole('rowgroup').map((group) =>
        within(group)
          .queryAllByText(/ answer \d$/)
          .map((cell) => cell.textContent),
      ),
    ).toEqual([
      ['baseline answer 2', 'candidate answer 2'],
      ['baseline answer 1', 'candidate answer 1'],
    ]);
  });

  test('waits for the recorded inputs before showing matched rows', async () => {
    let release = () => {};
    state.inputsReady = new Promise<void>((resolve) => {
      release = resolve;
    });
    renderTable();
    try {
      await waitFor(() => expect(state.inputsRequested).toBe(true), { timeout: 20000 });
      expect(screen.getByRole('region', { name: 'Traces' })).toHaveAttribute('aria-busy', 'true');
      expect(screen.queryByText('baseline answer to First question')).not.toBeInTheDocument();
    } finally {
      release();
    }
    expect(await findMatchedRows()).toBeVisible();
  });

  test('retries a failed run search into a complete comparison', async () => {
    state.failRun = 'candidate';
    renderTable();
    expect(await screen.findByText("Couldn't load traces", undefined, { timeout: 20000 })).toBeInTheDocument();
    state.failRun = undefined;
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(await findMatchedRows()).toBeVisible();
  });

  test('switches the empty state to no results only once a search is submitted', async () => {
    state.tracesByRun = {};
    renderTable();
    expect(await screen.findByText('No traces in this evaluation run', undefined, { timeout: 20000 })).toBeVisible();
    const search = screen.getByRole('textbox', { name: 'Search traces by id or input' });
    await userEvent.type(search, 'missing');
    expect(screen.getByText('No traces in this evaluation run')).toBeVisible();
    fireEvent.keyDown(search, { key: 'Enter', code: 'Enter', keyCode: 13, which: 13 });
    expect(await screen.findByText('No traces match your filters')).toBeVisible();
  });

  test('warns that the comparison is incomplete when a run reaches the trace limit', async () => {
    state.tracesByRun = {
      baseline: Array.from({ length: 1000 }, (_, index) =>
        makeRunTrace(traceId(`b${index.toString(16)}`), 'baseline', { request_preview: `Input ${index}` }),
      ),
      candidate: [],
    };
    renderTable();
    expect(
      await screen.findByText(
        'Comparison is limited to 1,000 traces per run. Matches and results may be incomplete.',
        undefined,
        {
          timeout: 30000,
        },
      ),
    ).toBeVisible();
  });

  test('leaves Time and Run name out of the default columns', async () => {
    renderTable();
    await findMatchedRows();
    expect(screen.queryByRole('columnheader', { name: /^Time/ })).not.toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: /^Run name/ })).not.toBeInTheDocument();
    expect(
      screen
        .getAllByRole('columnheader')
        .slice(0, 3)
        .map((header) => header.textContent),
    ).toEqual(['Input', 'Run', 'Output']);
  });

  test('offers only the comparison filter fields and no session grouping or date range in the toolbar', async () => {
    renderTable();
    await findMatchedRows();
    expect(screen.queryByRole('button', { name: 'Group traces by session' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /time range|last 7 days|all time/i })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /Filters/ }));
    await userEvent.click(await screen.findByRole('combobox', { name: /field/i }));
    expect(await screen.findByRole('option', { name: 'State' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Trace name' })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Run name' })).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Span name' })).not.toBeInTheDocument();
  });
});
