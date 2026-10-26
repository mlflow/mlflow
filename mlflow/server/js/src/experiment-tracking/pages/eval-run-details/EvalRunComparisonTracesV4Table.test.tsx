import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { configure, fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useEffect } from 'react';
import { rest } from 'msw';
import {
  ASSESSMENT_SESSION_METADATA_KEY,
  SESSION_ID_METADATA_KEY,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';
import { setupServer } from '../../../common/utils/setup-msw';
import { setupTestRouter } from '../../../common/utils/RoutingTestUtils';
import { useLocation } from '../../../common/utils/RoutingUtils';
import { renderTracesV4Page } from '../../components/experiment-page/components/traces-v4/test-utils/renderTracesV4Page';
import {
  makeFeedbackAssessment,
  makeTrace,
} from '../../components/experiment-page/components/traces-v4/test-utils/mockTraces';
import {
  EvalRunComparisonTracesV4Table,
  type EvalRunComparisonTracesV4TableProps,
} from './EvalRunComparisonTracesV4Table';

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

  const renderTable = (initialUrl = COMPARE_URL, props: Partial<EvalRunComparisonTracesV4TableProps> = {}) =>
    renderTracesV4Page({
      initialUrl,
      routes: [
        {
          path: PATH,
          element: (
            <>
              <LocationSpy />
              <EvalRunComparisonTracesV4Table
                experimentId={EXPERIMENT_ID}
                runUuids={['baseline', 'candidate']}
                {...props}
              />
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

  test('explains comparison filtering and keeps an incomplete filter in the editor', async () => {
    renderTable();
    expect(await findMatchedRows()).toBeVisible();
    const [filters, ...otherFilterButtons] = screen.getAllByRole('button', { name: /Filters/ });
    expect(otherFilterButtons).toHaveLength(0);
    await userEvent.click(filters);
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
    expect(
      screen.getByText('Could not load traces for all selected runs. Retry to load the comparison.'),
    ).toBeVisible();
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

  test('offers only the comparison filter fields, session grouping and no date range in the toolbar', async () => {
    renderTable();
    await findMatchedRows();
    expect(screen.getByRole('radio', { name: 'Traces' })).toBeChecked();
    expect(screen.queryByRole('button', { name: /time range|last 7 days|all time/i })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /Filters/ }));
    await userEvent.click(await screen.findByRole('combobox', { name: /field/i }));
    expect(await screen.findByRole('option', { name: 'State' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Trace name' })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Run name' })).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Span name' })).not.toBeInTheDocument();
  });

  // Two runs' versions of one simulated conversation: baseline has two turns, candidate one. Each turn has a
  // trace-level quality score; each session has a session-level "Conversation quality" score.
  const setupSessionTraces = () => {
    const turns = (runUuid: string, count: number, quality: number[], conversation: number) =>
      Array.from({ length: count }, (_, turn) => {
        const sessionId = `${runUuid}-conversation`;
        // Hex ids pass the exact trace-id check, so a turn can open in the drawer.
        return makeRunTrace(traceId(`${runUuid === 'baseline' ? 'ba' : 'ca'}${turn}`), runUuid, {
          request_time: new Date(Date.UTC(2026, 0, 1, 0, turn)).toISOString(),
          request_preview: `${runUuid} input ${turn}`,
          response_preview: `${runUuid} output ${turn}`,
          trace_metadata: {
            'mlflow.sourceRun': runUuid,
            [SESSION_ID_METADATA_KEY]: sessionId,
            'mlflow.simulation.goal': 'Plan a trip',
            'mlflow.simulation.persona': 'Traveler',
          },
          assessments: [
            makeFeedbackAssessment('quality', quality[turn], {
              metadata: { 'mlflow.assessment.sourceRunId': runUuid },
            }),
            makeFeedbackAssessment('Conversation quality', conversation, {
              metadata: { [ASSESSMENT_SESSION_METADATA_KEY]: sessionId, 'mlflow.assessment.sourceRunId': runUuid },
            }),
          ],
        });
      });
    state.tracesByRun = {
      baseline: turns('baseline', 2, [0.2, 0.6], 0.9),
      candidate: turns('candidate', 1, [0.8], 1),
    };
    return state.tracesByRun;
  };
  const sessionsView = () => screen.findByRole('radio', { name: 'Sessions' });
  const tracesView = () => screen.findByRole('radio', { name: 'Traces' });

  test('switches to nested session summaries, expands runs independently and opens their traces', async () => {
    setupSessionTraces();
    renderTable();
    await screen.findByText('3 comparison rows');
    expect(screen.queryByRole('columnheader', { name: /^Conversation quality/ })).not.toBeInTheDocument();

    await userEvent.click(await sessionsView());

    expect(await screen.findByRole('button', { name: 'Collapse session comparison Matched sessions' })).toBeVisible();
    expect(new URLSearchParams(lastSearch).get('comparisonGroupBy')).toBe('session');
    expect(screen.getByRole('radio', { name: 'Sessions' })).toBeChecked();
    expect(screen.queryByRole('columnheader', { name: 'Run' })).not.toBeInTheDocument();
    expect(screen.getAllByRole('columnheader')[0]).toHaveTextContent(/^Session/);
    const groupRow = within(screen.getByRole('row', { name: 'Session comparison Matched sessions' }));
    expect(groupRow.getByText('2 runs · 3 turns')).toBeVisible();
    const quality = await screen.findByRole('columnheader', { name: /^quality/ });
    await waitFor(() => expect(within(quality).getByRole('group', { name: 'Baseline run' })).toHaveTextContent('0.4'));
    expect(within(quality).getByRole('group', { name: 'Baseline run' })).toHaveTextContent('AVG');
    expect(within(quality).getByRole('group', { name: 'Candidate run' })).toHaveTextContent('0.8');
    expect(within(quality).getByRole('group', { name: 'Candidate run' })).toHaveTextContent('+0.4');
    const conversation = screen.getByRole('columnheader', { name: /^Conversation quality/ });
    expect(within(conversation).getByRole('group', { name: 'Baseline run' })).toHaveTextContent('0.9');

    const baselineSummary = within(screen.getByRole('row', { name: 'Baseline run session baseline-conversation' }));
    const candidateSummary = within(screen.getByRole('row', { name: 'Candidate run session candidate-conversation' }));
    expect(baselineSummary.getByText('baseline input 0')).toBeVisible();
    expect(baselineSummary.getByText('baseline output 1')).toBeVisible();
    expect(baselineSummary.getByText('0.4 (AVG)')).toBeVisible();
    expect(candidateSummary.getByText('candidate output 0')).toBeVisible();
    expect(candidateSummary.getByText('0.8 (AVG)')).toBeVisible();
    expect(baselineSummary.getByRole('link', { name: 'baseline-conversation' })).toHaveAttribute(
      'href',
      expect.stringContaining('baseline-conversation'),
    );
    expect(screen.queryByRole('rowgroup', { name: 'Baseline run session turns' })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Expand Baseline run session baseline-conversation' }));
    const baselineTurns = within(screen.getByRole('rowgroup', { name: 'Baseline run session turns' }));
    expect(baselineTurns.getByText('Turn 1')).toBeVisible();
    expect(baselineTurns.getByText('Turn 2')).toBeVisible();
    expect(screen.queryByRole('rowgroup', { name: 'Candidate run session turns' })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Expand Candidate run session candidate-conversation' }));
    const candidateTurns = within(screen.getByRole('rowgroup', { name: 'Candidate run session turns' }));
    expect(candidateTurns.queryByText('Turn 2')).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Collapse Baseline run session baseline-conversation' }));
    expect(screen.queryByRole('rowgroup', { name: 'Baseline run session turns' })).not.toBeInTheDocument();
    expect(screen.getByRole('rowgroup', { name: 'Candidate run session turns' })).toBeVisible();
    expect(screen.getByText('1 comparison group')).toBeVisible();

    await userEvent.click(candidateTurns.getByText('candidate output 0'));
    await waitFor(() => expect(new URLSearchParams(lastSearch).get('selectedEvaluationId')).toBe(traceId('ca0')));
    expect(new URLSearchParams(lastSearch).get('comparisonGroupBy')).toBe('session');
  }, 60000);

  test('keeps mode column preferences separate while resetting both expansion levels on grouping round trips', async () => {
    setupSessionTraces();
    renderTable();
    const duration = await screen.findByRole('columnheader', { name: /^Duration/ });
    await userEvent.click(within(duration).getByRole('button', { name: /Column options/ }));
    await userEvent.click(screen.getByRole('menuitem', { name: 'Hide column' }));
    expect(screen.queryByRole('columnheader', { name: /^Duration/ })).not.toBeInTheDocument();

    await userEvent.click(await sessionsView());
    expect(await screen.findByRole('columnheader', { name: /^Duration/ })).toBeVisible();
    const quality = screen.getByRole('columnheader', { name: /^quality/ });
    await userEvent.click(within(quality).getByRole('button', { name: /Column options/ }));
    await userEvent.click(screen.getByRole('menuitem', { name: 'Hide column' }));
    expect(screen.queryByRole('columnheader', { name: /^quality/ })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Expand Baseline run session baseline-conversation' }));
    expect(screen.getByRole('rowgroup', { name: 'Baseline run session turns' })).toBeVisible();
    await userEvent.click(screen.getByRole('button', { name: 'Collapse session comparison Matched sessions' }));
    expect(screen.queryByRole('row', { name: 'Baseline run session baseline-conversation' })).not.toBeInTheDocument();

    await userEvent.click(await tracesView());
    expect(await screen.findByRole('columnheader', { name: /^quality/ })).toBeVisible();
    expect(screen.queryByRole('columnheader', { name: /^Duration/ })).not.toBeInTheDocument();
    expect(screen.getAllByRole('columnheader')[0]).toHaveTextContent('Input');

    await userEvent.click(await sessionsView());
    expect(await screen.findByRole('columnheader', { name: /^Duration/ })).toBeVisible();
    expect(screen.queryByRole('columnheader', { name: /^quality/ })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Collapse session comparison Matched sessions' })).toHaveAttribute(
      'aria-expanded',
      'true',
    );
    expect(screen.getByRole('button', { name: 'Expand Baseline run session baseline-conversation' })).toHaveAttribute(
      'aria-expanded',
      'false',
    );
    expect(screen.queryByRole('rowgroup', { name: 'Baseline run session turns' })).not.toBeInTheDocument();
  }, 60000);

  test('keeps an Overview outcome filter removed when switching to sessions', async () => {
    setupSessionTraces();
    renderTable(COMPARE_URL, {
      initialOutcomeFilter: { columnId: 'assessment:quality', runUuid: 'candidate', outcome: 'improvement' },
    });
    await userEvent.click(await screen.findByRole('button', { name: /Remove comparison filter/ }));
    await userEvent.click(await sessionsView());

    expect(await screen.findByRole('button', { name: 'Collapse session comparison Matched sessions' })).toBeVisible();
    expect(new URLSearchParams(lastSearch).get('comparisonGroupBy')).toBe('session');
    expect(screen.getByRole('radio', { name: 'Sessions' })).toBeChecked();
  }, 60000);

  test('locks the Session, Input and Output columns while comparing sessions', async () => {
    setupSessionTraces();
    renderTable(`${COMPARE_URL}&comparisonGroupBy=session`);
    await screen.findByRole('button', { name: 'Expand Baseline run session baseline-conversation' });
    for (const name of [/^Session/, /^Input/, /^Output/]) {
      const header = screen.getByRole('columnheader', { name });
      expect(within(header).queryByRole('button', { name: /Column options/ })).not.toBeInTheDocument();
    }
    const duration = screen.getByRole('columnheader', { name: /^Duration/ });
    expect(within(duration).getByRole('button', { name: /Column options/ })).toBeVisible();

    await userEvent.click(screen.getByRole('button', { name: 'Display' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Columns' }));
    for (const name of ['Session', 'Input', 'Output']) {
      expect(await screen.findByRole('menuitemcheckbox', { name })).toHaveAttribute('aria-disabled', 'true');
    }
    expect(screen.getByRole('menuitemcheckbox', { name: 'Duration' })).not.toHaveAttribute('aria-disabled');
  }, 60000);

  test('explains that session comparisons filter on individual traces', async () => {
    setupSessionTraces();
    renderTable(`${COMPARE_URL}&comparisonGroupBy=session`);
    await screen.findByRole('button', { name: 'Expand Baseline run session baseline-conversation' });
    await userEvent.click(screen.getByRole('button', { name: /Filters/ }));
    expect(
      screen.getByText(
        'Keep every turn and run in a session comparison when one trace matches all filters and the search. Filters use individual trace values, not session scores or totals.',
      ),
    ).toBeVisible();
  });

  test('keeps failed trace feedback visible when expanded without counting it in the session summary', async () => {
    const traces = setupSessionTraces();
    traces['baseline'][0].assessments?.push(
      makeFeedbackAssessment('quality', 100, {
        create_time: '2026-02-01T00:00:00Z',
        error: { error_code: 'FAILED', error_message: 'Scorer unavailable' },
        metadata: { 'mlflow.assessment.sourceRunId': 'baseline' },
      }),
    );
    renderTable(`${COMPARE_URL}&comparisonGroupBy=session`);
    const quality = await screen.findByRole('columnheader', { name: /^quality/ });
    await waitFor(() => expect(within(quality).getByRole('group', { name: 'Baseline run' })).toHaveTextContent('0.6'));
    await userEvent.click(
      await screen.findByRole('button', { name: 'Expand Baseline run session baseline-conversation' }),
    );
    const baselineTurns = within(screen.getByRole('rowgroup', { name: 'Baseline run session turns' }));
    expect(baselineTurns.getByText('Error')).toBeVisible();
  });

  test('uses one session-scoped scorer column across runs without substituting missing session scores', async () => {
    const traces = setupSessionTraces();
    traces['baseline'][0].assessments?.push(
      makeFeedbackAssessment('quality', 0.95, {
        metadata: {
          [ASSESSMENT_SESSION_METADATA_KEY]: 'baseline-conversation',
          'mlflow.assessment.sourceRunId': 'baseline',
        },
      }),
    );
    renderTable(`${COMPARE_URL}&comparisonGroupBy=session`);
    const quality = await screen.findByRole('columnheader', { name: /^quality/ });
    await waitFor(() => expect(within(quality).getByRole('group', { name: 'Baseline run' })).toHaveTextContent('0.95'));
    expect(within(quality).getByRole('group', { name: 'Candidate run' })).toHaveTextContent('—');
    const candidateSummary = within(screen.getByRole('row', { name: 'Candidate run session candidate-conversation' }));
    expect(candidateSummary.queryByText('0.8 (AVG)')).not.toBeInTheDocument();
  });

  test('hides session grouping while an outcome filter is active and leaves session mode for it', async () => {
    renderTable();
    await findMatchedRows();
    expect(await sessionsView()).toBeVisible();
    const qualityHeader = await screen.findByRole('columnheader', { name: /quality/ });
    const candidate = await waitFor(() => within(qualityHeader).getByRole('group', { name: 'Candidate run' }));
    await userEvent.click(within(candidate).getByRole('button', { name: /Open comparison summary/ }));
    await userEvent.click(await screen.findByTestId('comparison-summary-improvement-filter'));
    expect(await screen.findByRole('button', { name: /Edit comparison filter/ })).toBeVisible();
    expect(screen.queryByRole('radio', { name: 'Sessions' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /Remove comparison filter/ }));
    expect(await sessionsView()).toBeVisible();
  });
});
