import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useEffect } from 'react';
import { rest } from 'msw';
import { setupServer } from '../../../common/utils/setup-msw';
import { setupTestRouter } from '../../../common/utils/RoutingTestUtils';
import { useLocation } from '../../../common/utils/RoutingUtils';
import { renderTracesV4Page } from '../../components/experiment-page/components/traces-v4/test-utils/renderTracesV4Page';
import {
  makeFeedbackAssessment,
  makeTrace,
  makeTraces,
} from '../../components/experiment-page/components/traces-v4/test-utils/mockTraces';
import { EvalRunTracesV4Table } from './EvalRunTracesV4Table';

const EXPERIMENT_ID = 'exp-1';
const RUN_UUID = 'run-1';
const PATH = '/ml/experiments/:experimentId/evaluation-runs/:runUuid/details';
const RUN_URL = `/ml/experiments/${EXPERIMENT_ID}/evaluation-runs/${RUN_UUID}/details`;
const SEARCH_ENDPOINT = '/ajax-api/3.0/mlflow/traces/search';
// OSS traces live in the experiment, so their URL id is the bare trace id.
const makeRunTrace = (...[id, over]: Parameters<typeof makeTrace>) =>
  makeTrace(id, {
    trace_location: { type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: EXPERIMENT_ID } },
    ...over,
  });
const makeRunTraces = (count: number) => makeTraces(count).map((trace) => makeRunTrace(trace.trace_id));
const RUN_FILTER = `attributes.run_id = '${RUN_UUID}' AND tags.\`mlflow.trace.sourceScorer\` IS NULL`;

interface SearchCall {
  filter?: string;
  max_results?: number;
  locations?: { type?: string; mlflow_experiment?: { experiment_id?: string } }[];
}

const state: {
  traces: ReturnType<typeof makeRunTraces>;
  exactTraces: ReturnType<typeof makeRunTraces>;
  searchCalls: SearchCall[];
} = { traces: [], exactTraces: [], searchCalls: [] };
let lastSearch = '';

const server = setupServer(
  rest.post(SEARCH_ENDPOINT, async (req, res, ctx) => {
    const body = (await req.json()) as SearchCall;
    state.searchCalls.push(body);
    const traces = body.filter?.includes('attributes.request_id') ? state.exactTraces : state.traces;
    return res(ctx.json({ traces, next_page_token: undefined }));
  }),
  rest.post('/ajax-api/3.0/mlflow/datasets/search', (_req, res, ctx) => res(ctx.json({ datasets: [] }))),
);

const LocationSpy = () => {
  const search = useLocation().search;
  useEffect(() => {
    lastSearch = search;
  }, [search]);
  return null;
};

describe('EvalRunTracesV4Table', () => {
  const { history } = setupTestRouter();

  const renderTable = (initialUrl = RUN_URL) =>
    renderTracesV4Page({
      initialUrl,
      routes: [
        {
          path: PATH,
          element: (
            <>
              <LocationSpy />
              <EvalRunTracesV4Table experimentId={EXPERIMENT_ID} runUuid={RUN_UUID} />
            </>
          ),
        },
      ],
      history,
      experimentId: EXPERIMENT_ID,
    });

  const findTraceRow = (traceId: string) => screen.findByLabelText(`Open trace ${traceId} — input`, { selector: 'a' });

  beforeAll(() => {
    process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
    server.listen();
  });

  beforeEach(() => {
    window.localStorage.setItem('mlflow.detectIssues.guidanceShown_v1', 'true');
    state.traces = makeRunTraces(3);
    state.exactTraces = [];
    state.searchCalls = [];
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

  test('searches only the run’s traces in the experiment, without a time range', async () => {
    renderTable();

    expect(await findTraceRow('tr-000')).toBeInTheDocument();
    const firstCall = state.searchCalls[0];
    expect(firstCall.filter).toBe(RUN_FILTER);
    expect(firstCall.locations).toEqual([{ type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'exp-1' } }]);
    expect(screen.queryByRole('button', { name: /last 7 days|all time|time range/i })).not.toBeInTheDocument();
  });

  test('describes an empty evaluation run', async () => {
    state.traces = [];
    renderTable();

    expect(await screen.findByText('No traces in this evaluation run')).toBeInTheDocument();
  });

  test('shows only the assessments this run produced', async () => {
    state.traces = [
      makeRunTrace('tr-000', {
        assessments: [
          makeFeedbackAssessment('relevance', true, { metadata: { 'mlflow.assessment.sourceRunId': RUN_UUID } }),
          makeFeedbackAssessment('other_run_judge', true, { metadata: { 'mlflow.assessment.sourceRunId': 'run-2' } }),
        ],
      }),
    ];
    renderTable();

    expect(await screen.findByRole('columnheader', { name: /relevance/ })).toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: /other_run_judge/ })).not.toBeInTheDocument();
  });

  test('links rows to the run view with the legacy drawer parameter', async () => {
    renderTable(`${RUN_URL}?traceId=tr-old&spanId=span-1`);

    const row = await findTraceRow('tr-001');
    const href = new URL(row.getAttribute('href') ?? '', 'http://localhost');
    expect(href.pathname).toBe(RUN_URL);
    expect(href.searchParams.get('selectedEvaluationId')).toBe('tr-001');
    expect(href.searchParams.get('traceId')).toBeNull();
    expect(href.searchParams.get('spanId')).toBeNull();
  });

  test('clears a deep link to a trace that is not part of the run', async () => {
    const outsideTraceId = 'tr-0123456789abcdef0123456789abcdef';
    renderTable(`${RUN_URL}?selectedEvaluationId=${outsideTraceId}`);

    await findTraceRow('tr-000');
    await waitFor(() => expect(new URLSearchParams(lastSearch).get('selectedEvaluationId')).toBeNull());
    const exactCall = state.searchCalls.find((call) => call.filter?.includes('attributes.request_id'));
    expect(exactCall?.filter).toBe(`${RUN_FILTER} AND attributes.request_id = '${outsideTraceId}'`);
  });

  test('clears a malformed deep link without a lookup', async () => {
    renderTable(`${RUN_URL}?selectedEvaluationId=not-a-trace-id`);

    await findTraceRow('tr-000');
    await waitFor(() => expect(new URLSearchParams(lastSearch).get('selectedEvaluationId')).toBeNull());
    expect(state.searchCalls.some((call) => call.filter?.includes('attributes.request_id'))).toBe(false);
  });

  test('hides Request time and never offers the Run column', async () => {
    const user = userEvent.setup();
    renderTable();

    await findTraceRow('tr-000');
    expect(screen.queryByRole('columnheader', { name: 'Time' })).not.toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: /^Run/ })).not.toBeInTheDocument();

    await user.click(screen.getByLabelText('Display', { selector: 'button' }));
    const menu = await screen.findByRole('menu', { name: 'Display' });
    await user.click(within(menu).getByRole('menuitem', { name: /Columns/ }));
    expect(await screen.findByRole('menuitemcheckbox', { name: /Time/ })).not.toBeChecked();
    expect(screen.queryByRole('menuitemcheckbox', { name: /^Run/ })).not.toBeInTheDocument();
  });

  test('turns a typed query into a filter on the run’s traces', async () => {
    state.traces = [makeRunTrace('tr-000', { assessments: [makeFeedbackAssessment('relevance', 'yes')] })];
    renderTable();

    await findTraceRow('tr-000');
    expect(screen.queryByRole('button', { name: 'Filters' })).not.toBeInTheDocument();
    const searchBox = screen.getByRole('combobox', { name: 'Type something to use full text search' });
    expect(searchBox).toHaveAttribute('placeholder', 'Add a filter');

    fireEvent.change(searchBox, { target: { value: "feedback.relevance = 'yes'" } });
    fireEvent.keyDown(searchBox, { key: 'Enter', code: 'Enter', keyCode: 13, which: 13 });

    // Feedback filters match the run's own assessments on the page, so the server only scopes to the run.
    await waitFor(() => expect(state.searchCalls[state.searchCalls.length - 1]?.max_results).toBe(500));
    expect(state.searchCalls[state.searchCalls.length - 1]?.filter).toBe(RUN_FILTER);
    expect(await screen.findByRole('button', { name: "Edit filter feedback.relevance = 'yes'" })).toBeInTheDocument();
    expect(await findTraceRow('tr-000')).toBeInTheDocument();
  });

  test('pages through the feedback matches on the page', async () => {
    const ofThisRun = { metadata: { 'mlflow.assessment.sourceRunId': RUN_UUID } };
    state.traces = Array.from({ length: 30 }, (_, i) =>
      makeRunTrace(`tr-${String(i).padStart(3, '0')}`, {
        assessments: [makeFeedbackAssessment('correctness', i < 27 ? 'yes' : 'no', ofThisRun)],
      }),
    );
    renderTable();
    await findTraceRow('tr-000');

    const searchBox = screen.getByRole('combobox', { name: 'Type something to use full text search' });
    fireEvent.change(searchBox, { target: { value: "feedback.correctness = 'yes'" } });
    fireEvent.keyDown(searchBox, { key: 'Enter', code: 'Enter', keyCode: 13, which: 13 });
    await screen.findByRole('button', { name: "Edit filter feedback.correctness = 'yes'" });
    expect(await findTraceRow('tr-024')).toBeInTheDocument();

    // 27 matches over pages of 25: the second page holds the last 2.
    await userEvent.click(screen.getByRole('button', { name: 'Next page' }));
    expect(await findTraceRow('tr-026')).toBeInTheDocument();
    expect(await findTraceRow('tr-025')).toBeInTheDocument();
    expect(screen.queryByText(/tr-024/)).not.toBeInTheDocument();
    expect(screen.queryByText(/tr-027/)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Next page' })).toBeDisabled();
  }, 60000);

  test("filters linked traces on the run's own feedback, not another run's", async () => {
    const ofRun = (runUuid: string) => ({ metadata: { 'mlflow.assessment.sourceRunId': runUuid } });
    state.traces = [
      // Linked to this run and another: the other run said "no", this run said "yes".
      makeRunTrace('tr-linked', {
        assessments: [
          makeFeedbackAssessment('correctness', 'no', ofRun('run-2')),
          makeFeedbackAssessment('correctness', 'yes', ofRun(RUN_UUID)),
        ],
      }),
      makeRunTrace('tr-own-no', { assessments: [makeFeedbackAssessment('correctness', 'no', ofRun(RUN_UUID))] }),
    ];
    renderTable();
    await findTraceRow('tr-linked');

    const searchBox = screen.getByRole('combobox', { name: 'Type something to use full text search' });
    fireEvent.change(searchBox, { target: { value: "feedback.correctness = 'no'" } });
    fireEvent.keyDown(searchBox, { key: 'Enter', code: 'Enter', keyCode: 13, which: 13 });

    expect(await screen.findByRole('button', { name: "Edit filter feedback.correctness = 'no'" })).toBeInTheDocument();
    expect(await findTraceRow('tr-own-no')).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByText(/tr-linked/)).not.toBeInTheDocument());
  });

  describe('assessment summaries in the column headers', () => {
    const ofRun = { metadata: { 'mlflow.assessment.sourceRunId': RUN_UUID } };
    const summaryTraces = () => [
      makeRunTrace('tr-000', { assessments: [makeFeedbackAssessment('correctness', 'yes', ofRun)] }),
      makeRunTrace('tr-001', { assessments: [makeFeedbackAssessment('correctness', 'yes', ofRun)] }),
      makeRunTrace('tr-002', { assessments: [makeFeedbackAssessment('correctness', 'no', ofRun)] }),
      makeRunTrace('tr-003', { assessments: [makeFeedbackAssessment('correctness', 'yes', ofRun)] }),
    ];

    test('summarizes each scorer over the whole run in its header', async () => {
      state.traces = summaryTraces();
      renderTable();

      const header = await screen.findByRole('columnheader', { name: /^Correctness PASS 75% / });
      expect(within(header).getByRole('progressbar', { name: 'Pass' })).toHaveAttribute('aria-valuenow', '75');
      // The summaries search the whole run, independently of the table's page and filters.
      expect(state.searchCalls.filter((call) => call.filter === RUN_FILTER).length).toBeGreaterThanOrEqual(2);
    });

    test('filters the table on a value clicked in a header chart, and clears it on a second click', async () => {
      state.traces = summaryTraces();
      renderTable();

      const header = await screen.findByRole('columnheader', { name: /^Correctness PASS 75% / });
      await userEvent.click(within(header).getByRole('progressbar', { name: 'Fail' }));
      await waitFor(() =>
        expect(state.searchCalls[state.searchCalls.length - 1]?.filter).toBe(
          `${RUN_FILTER} AND feedback.\`correctness\` = 'no'`,
        ),
      );

      const pill = await screen.findByRole('button', { name: "Edit filter feedback.correctness = 'no'" });
      expect(pill).toBeInTheDocument();

      await userEvent.click(
        within(screen.getByRole('columnheader', { name: /^Correctness PASS 75% / })).getByRole('progressbar', {
          name: 'Fail',
        }),
      );
      await waitFor(() =>
        expect(
          screen.queryByRole('button', { name: "Edit filter feedback.correctness = 'no'" }),
        ).not.toBeInTheDocument(),
      );
    });

    test('collapses and expands the header charts', async () => {
      state.traces = summaryTraces();
      renderTable();

      const header = await screen.findByRole('columnheader', { name: /^Correctness PASS 75% / });
      expect(within(header).getAllByRole('progressbar').length).toBeGreaterThan(0);

      await userEvent.click(screen.getByRole('button', { name: 'Collapse assessment charts' }));
      expect(within(screen.getByRole('columnheader', { name: /^Correctness/ })).queryByRole('progressbar')).toBeNull();
      expect(screen.getByRole('columnheader', { name: /^Correctness/ })).toHaveAccessibleName('Correctness PASS 75%');

      await userEvent.click(screen.getByRole('button', { name: 'Expand assessment charts' }));
      expect(
        within(screen.getByRole('columnheader', { name: /^Correctness/ })).getAllByRole('progressbar').length,
      ).toBeGreaterThan(0);
    });

    test('shows no summary toggle when no scorer has a chart', async () => {
      state.traces = [makeRunTrace('tr-000', { assessments: [makeFeedbackAssessment('relevance', 4, ofRun)] })];
      renderTable();

      expect(await screen.findByRole('columnheader', { name: 'relevance AVG 4' })).toBeInTheDocument();
      expect(screen.queryByRole('button', { name: /assessment charts/ })).not.toBeInTheDocument();
    });
  });
});
