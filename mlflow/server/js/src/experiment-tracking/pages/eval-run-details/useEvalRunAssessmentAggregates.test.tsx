import { afterAll, beforeAll, beforeEach, describe, expect, test } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import { rest } from 'msw';
import { IntlProvider } from 'react-intl';
import { setupServer } from '../../../common/utils/setup-msw';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import {
  makeFeedbackAssessment,
  makeTrace,
} from '../../components/experiment-page/components/traces-v4/test-utils/mockTraces';
import { AGGREGATE_TRACE_LIMIT, useEvalRunAssessmentAggregates } from './useEvalRunAssessmentAggregates';

const RUN_UUID = 'run-1';
const RUN_FILTER = `attributes.run_id = '${RUN_UUID}' AND tags.\`mlflow.trace.sourceScorer\` IS NULL`;
const ofRun = (runUuid: string) => ({ metadata: { 'mlflow.assessment.sourceRunId': runUuid } });

const state: { pages: ReturnType<typeof makeTrace>[][]; calls: { filter?: string; page_token?: string }[] } = {
  pages: [],
  calls: [],
};
const server = setupServer(
  rest.post('/ajax-api/3.0/mlflow/traces/search', async (req, res, ctx) => {
    const body = (await req.json()) as { filter?: string; page_token?: string };
    state.calls.push(body);
    const index = body.page_token ? Number(body.page_token) : 0;
    const next = index + 1 < state.pages.length ? String(index + 1) : undefined;
    return res(ctx.json({ traces: state.pages[index] ?? [], next_page_token: next }));
  }),
);

const renderAggregates = () =>
  renderHook(() => useEvalRunAssessmentAggregates({ experimentId: 'exp-1', runUuid: RUN_UUID }), {
    wrapper: ({ children }) => (
      <IntlProvider locale="en">
        <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
          {children}
        </QueryClientProvider>
      </IntlProvider>
    ),
  });

describe('useEvalRunAssessmentAggregates', () => {
  beforeAll(() => {
    process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
    server.listen();
  });
  beforeEach(() => {
    state.pages = [];
    state.calls = [];
  });
  afterAll(() => {
    delete process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'];
    server.close();
  });

  test('summarizes each scorer over the whole run, from this run’s assessments only', async () => {
    state.pages = [
      [
        makeTrace('tr-1', {
          assessments: [
            makeFeedbackAssessment('correctness', 'yes', ofRun(RUN_UUID)),
            makeFeedbackAssessment('is_polite', true, ofRun(RUN_UUID)),
            makeFeedbackAssessment('relevance', 4, ofRun(RUN_UUID)),
            makeFeedbackAssessment('other_run_judge', 'yes', ofRun('run-2')),
          ],
        }),
        makeTrace('tr-2', {
          assessments: [
            makeFeedbackAssessment('correctness', 'no', ofRun(RUN_UUID)),
            makeFeedbackAssessment('is_polite', true, ofRun(RUN_UUID)),
            makeFeedbackAssessment('relevance', 2, ofRun(RUN_UUID)),
          ],
        }),
      ],
    ];
    const { result } = renderAggregates();

    await waitFor(() => expect(result.current.scorers).toHaveLength(3));
    expect(state.calls[0].filter).toBe(RUN_FILTER);
    const byName = new Map(result.current.scorers.map((scorer) => [scorer.name, scorer]));
    expect(byName.has('other_run_judge')).toBe(false);
    expect(byName.get('correctness')?.dtype).toBe('pass-fail');
    expect(byName.get('correctness')?.aggregates.currentCounts?.get('yes')).toBe(1);
    expect(byName.get('correctness')?.aggregates.currentCounts?.get('no')).toBe(1);
    expect(byName.get('is_polite')?.dtype).toBe('boolean');
    expect(byName.get('is_polite')?.aggregates.currentCounts?.get(true)).toBe(2);
    expect(byName.get('relevance')?.dtype).toBe('numeric');
    expect(byName.get('relevance')?.aggregates.currentNumericAggregate?.average).toBe(3);
  });

  test('summarizes a session-level scorer over the sessions, not every turn', async () => {
    const ofSession = (sessionId: string) => ({
      metadata: { 'mlflow.assessment.sourceRunId': RUN_UUID, 'mlflow.trace.session': sessionId },
    });
    // Two sessions of two turns; the session judges sit on each session's first turn.
    state.pages = [
      [
        makeTrace('s1-t1', {
          assessments: [
            makeFeedbackAssessment('correctness', 'yes', ofRun(RUN_UUID)),
            makeFeedbackAssessment('completeness', 'yes', ofSession('s1')),
            makeFeedbackAssessment('frustration', 'none', ofSession('s1')),
          ],
        }),
        makeTrace('s1-t2', { assessments: [makeFeedbackAssessment('correctness', 'no', ofRun(RUN_UUID))] }),
        makeTrace('s2-t1', {
          assessments: [
            makeFeedbackAssessment('correctness', 'yes', ofRun(RUN_UUID)),
            makeFeedbackAssessment('completeness', 'no', ofSession('s2')),
            makeFeedbackAssessment('frustration', 'high', ofSession('s2')),
          ],
        }),
        makeTrace('s2-t2', { assessments: [makeFeedbackAssessment('correctness', 'yes', ofRun(RUN_UUID))] }),
      ],
    ];
    const { result } = renderAggregates();

    await waitFor(() => expect(result.current.scorers).toHaveLength(3));
    const byName = new Map(result.current.scorers.map((scorer) => [scorer.name, scorer]));
    const completeness = byName.get('completeness')?.aggregates.currentCounts;
    expect(completeness?.get('yes')).toBe(1);
    expect(completeness?.get('no')).toBe(1);
    expect(completeness?.get(undefined) ?? 0).toBe(0);
    const frustration = byName.get('frustration')?.aggregates.currentCounts;
    expect(
      [...(frustration?.entries() ?? [])]
        .filter(([, count]) => count > 0)
        .map(([value]) => value)
        .sort(),
    ).toEqual(['high', 'none']);
    expect([...(byName.get('frustration')?.assessmentInfo.uniqueValues ?? [])].sort()).toEqual(['high', 'none']);
    // Turn-level scorers still count every turn.
    expect(byName.get('correctness')?.aggregates.currentCounts?.get('yes')).toBe(3);
  });

  test(`pages through the run and stops at ${AGGREGATE_TRACE_LIMIT} traces`, async () => {
    const page = (start: number) =>
      Array.from({ length: 400 }, (_, i) =>
        makeTrace(`tr-${start + i}`, { assessments: [makeFeedbackAssessment('correctness', 'yes', ofRun(RUN_UUID))] }),
      );
    state.pages = [page(0), page(400), page(800), page(1200)];
    const { result } = renderAggregates();

    await waitFor(() => expect(result.current.scorers).toHaveLength(1));
    expect(state.calls).toHaveLength(3);
    expect(result.current.scorers[0].aggregates.currentCounts?.get('yes')).toBe(AGGREGATE_TRACE_LIMIT);
  });

  test('reports a failed search', async () => {
    server.use(
      rest.post('/ajax-api/3.0/mlflow/traces/search', (_req, res, ctx) =>
        res(ctx.status(500), ctx.json({ message: 'boom' })),
      ),
    );
    const { result } = renderAggregates();

    await waitFor(() => expect(result.current.error).not.toBeNull());
    expect(result.current.scorers).toEqual([]);
  });
});
