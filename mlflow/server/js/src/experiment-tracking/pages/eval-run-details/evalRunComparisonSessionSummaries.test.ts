import { describe, expect, it } from '@jest/globals';
import { createIntl } from 'react-intl';
import { KnownEvaluationResultAssessmentStringValue } from '@databricks/web-shared/genai-traces-table';
import {
  ASSESSMENT_SESSION_METADATA_KEY,
  type FeedbackAssessment,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';
import { NOTES_ASSESSMENT_NAME } from '@databricks/web-shared/model-trace-explorer/assessments-pane/AssessmentsPaneNotesSection';
import type { SessionComparisonRow, TracesComparisonRow } from '@databricks/web-shared/traces-table';
import {
  makeFeedbackAssessment,
  makeTrace,
} from '../../components/experiment-page/components/traces-v4/test-utils/mockTraces';
import {
  getEvalRunSessionComparisonColumnAggregates,
  getSessionAssessmentSummary,
  getSessionComparisonAssessmentInfos,
  getSessionComparisonSummaries,
  pickComparisonFeedback,
  sortSessionComparisonRows,
  type SessionComparisonAssessmentInfo,
} from './evalRunComparisonSessionSummaries';

const INTL = createIntl({ locale: 'en', messages: {} });
const NUMERIC_INFO: SessionComparisonAssessmentInfo = { name: 'quality', scope: 'trace', dtype: 'numeric' };
const sessionScore = (
  value: number | boolean | string,
  overrides: Partial<FeedbackAssessment> = {},
): FeedbackAssessment =>
  makeFeedbackAssessment('quality', value, {
    metadata: { [ASSESSMENT_SESSION_METADATA_KEY]: 'session' },
    ...overrides,
  });
const scoredTrace = (id: string, value: number | boolean | string): ModelTraceInfoV3 =>
  makeTrace(id, { assessments: [makeFeedbackAssessment('quality', value)] });
const makeRows = (tracesByRow: ModelTraceInfoV3[][][]): SessionComparisonRow[] =>
  tracesByRow.map((runs, index) => ({
    kind: 'session',
    id: `row-${index}`,
    runs: runs.map((traces, runIndex) => ({
      runUuid: `run-${runIndex}`,
      label: `Run ${runIndex}`,
      color: 'blue',
      sessionId: traces.length ? `session-${index}-${runIndex}` : undefined,
      traces,
    })),
  }));

// @ui-test-skill-generated
describe('pickComparisonFeedback', () => {
  it('selects latest eligible feedback of the requested scope with an assessment-ID tie-break', () => {
    const latest = sessionScore(3, { assessment_id: 'a', create_time: '2026-02-01T00:00:00Z' });
    const traces = [
      makeTrace('first', {
        assessments: [
          sessionScore(1),
          sessionScore(2, { assessment_id: 'z', create_time: latest.create_time }),
          latest,
          sessionScore(100, { valid: false, create_time: '2026-03-01T00:00:00Z' }),
          makeFeedbackAssessment('quality', 1000, { create_time: '2026-04-01T00:00:00Z' }),
          makeFeedbackAssessment('other', 2000, { create_time: '2026-04-01T00:00:00Z' }),
        ],
      }),
      makeTrace('second', { assessments: [latest] }),
    ];
    expect(pickComparisonFeedback(traces, 'quality', 'session')).toEqual(latest);
    expect(pickComparisonFeedback([...traces].reverse(), 'quality', 'session')).toEqual(latest);
    expect(pickComparisonFeedback(traces, 'missing', 'session')).toBeUndefined();
  });
});

// @ui-test-skill-generated
describe('getSessionComparisonAssessmentInfos', () => {
  it('keeps session scope from all loaded feedback but infers dtype from retained session values', () => {
    const retained = makeTrace('retained', {
      assessments: [makeFeedbackAssessment('quality', 'trace text'), sessionScore(0.5)],
    });
    const filtered = makeTrace('filtered', { assessments: [sessionScore('good')] });
    const traceOnly = scoredTrace('trace-only', 0.9);
    expect(
      getSessionComparisonAssessmentInfos([retained, filtered, traceOnly], makeRows([[[retained], [traceOnly]]])),
    ).toEqual([{ name: 'quality', scope: 'session', dtype: 'numeric' }]);
    expect(getSessionComparisonAssessmentInfos([retained, filtered, traceOnly], makeRows([[[traceOnly]]]))).toEqual([
      { name: 'quality', scope: 'session', dtype: 'string' },
    ]);
  });

  it('does not create columns for invalid, internal, or expectation assessments', () => {
    const trace = makeTrace('trace', {
      assessments: [
        makeFeedbackAssessment('quality', 0),
        makeFeedbackAssessment('invalid', 10, { valid: false }),
        makeFeedbackAssessment(NOTES_ASSESSMENT_NAME, 'note'),
        {
          assessment_id: 'expectation',
          assessment_name: 'expected-answer',
          trace_id: 'trace',
          source: { source_type: 'HUMAN', source_id: 'user' },
          create_time: '2026-01-01T00:00:00Z',
          last_update_time: '2026-01-01T00:00:00Z',
          expectation: { value: 'answer' },
        },
      ],
    });
    expect(getSessionComparisonAssessmentInfos([trace], makeRows([[[trace]]]))).toEqual([NUMERIC_INFO]);
  });
});

// @ui-test-skill-generated
describe('getSessionAssessmentSummary', () => {
  it('averages the latest eligible numeric sample per turn including zero and negative scores', () => {
    const latest = makeFeedbackAssessment('quality', 4, { create_time: '2026-01-01T00:00:00Z' });
    const zero = makeFeedbackAssessment('quality', 0);
    const negative = makeFeedbackAssessment('quality', -1);
    const traces = [
      makeTrace('first', { assessments: [makeFeedbackAssessment('quality', 100), latest] }),
      makeTrace('second', { assessments: [zero] }),
      makeTrace('third', { assessments: [negative] }),
    ];
    expect(getSessionAssessmentSummary(traces, NUMERIC_INFO)).toEqual({
      value: 1,
      values: [4, 0, -1],
      assessments: [latest, zero, negative],
      hasError: false,
    });
  });

  it('uses recorded session feedback once without substituting trace scores for missing session feedback', () => {
    const recorded = sessionScore(0.8);
    const traces = [
      makeTrace('first', { assessments: [makeFeedbackAssessment('quality', 1), recorded] }),
      makeTrace('second', { assessments: [recorded] }),
    ];
    const info: SessionComparisonAssessmentInfo = { ...NUMERIC_INFO, scope: 'session' };
    expect(getSessionAssessmentSummary(traces, info)).toEqual({
      value: 0.8,
      values: [0.8],
      assessments: [recorded],
      hasError: false,
    });
    expect(getSessionAssessmentSummary([scoredTrace('trace-only', 100)], info)).toEqual({
      value: undefined,
      values: [],
      assessments: [],
      hasError: false,
    });
  });

  it('retains error feedback for rendering while excluding errors and absent or nonfinite values from averages', () => {
    const failure = makeFeedbackAssessment('quality', 100, { error: { error_code: 'FAILED' } });
    const nestedFailure = makeFeedbackAssessment('quality', 100, { feedback: { error: { error_code: 'FAILED' } } });
    const traces = [
      makeTrace('error', { assessments: [failure] }),
      makeTrace('nested-error', { assessments: [nestedFailure] }),
      makeTrace('null', { assessments: [makeFeedbackAssessment('quality', 0, { feedback: { value: null } })] }),
      makeTrace('missing', { assessments: [makeFeedbackAssessment('quality', 0, { feedback: {} })] }),
      scoredTrace('nan', Number.NaN),
      scoredTrace('infinity', Infinity),
      scoredTrace('zero', 0),
    ];
    const summary = getSessionAssessmentSummary(traces, NUMERIC_INFO);
    expect(summary.value).toBe(0);
    expect(summary.values).toEqual([0]);
    expect(summary.hasError).toBe(true);
    expect(summary.assessments).toContainEqual(failure);
    expect(summary.assessments).toContainEqual(nestedFailure);
  });

  it.each([
    { dtype: 'boolean', values: [true, false, false], expected: 1 / 3 },
    {
      dtype: 'pass-fail',
      values: [KnownEvaluationResultAssessmentStringValue.YES, KnownEvaluationResultAssessmentStringValue.NO],
      expected: 0.5,
    },
    { dtype: 'string', values: ['good', 'bad', 'good'], expected: undefined },
  ] satisfies {
    dtype: SessionComparisonAssessmentInfo['dtype'];
    values: (number | boolean | string)[];
    expected: number | undefined;
  }[])('preserves $dtype samples and computes the corresponding session summary', ({ dtype, values, expected }) => {
    const traces = values.map((value, index) => scoredTrace(`trace-${index}`, value));
    const summary = getSessionAssessmentSummary(traces, { ...NUMERIC_INFO, dtype });
    expect(summary.values).toEqual(values);
    expect(summary.value).toBe(expected);
    expect(summary.hasError).toBe(false);
  });
});

// @ui-test-skill-generated
describe('getSessionComparisonSummaries', () => {
  it('uses first time, last state and execution duration/token/cost totals for each run', () => {
    const first = makeTrace('first', {
      request_time: '2026-01-01T00:00:00Z',
      execution_duration: '2s',
      state: 'ERROR',
      trace_metadata: {
        'mlflow.trace.tokenUsage': JSON.stringify({ total_tokens: 10 }),
        'mlflow.trace.cost': JSON.stringify({ total_cost: 0.25 }),
      },
    });
    const last = makeTrace('last', {
      request_time: '2026-01-01T01:00:00Z',
      execution_duration: '3s',
      state: 'OK',
      trace_metadata: {
        'mlflow.trace.tokenUsage': JSON.stringify({ total_tokens: 20 }),
        'mlflow.trace.cost': JSON.stringify({ total_cost: 0.75 }),
      },
    });
    expect(getSessionComparisonSummaries(makeRows([[[first, last]]]), []).get('row-0')).toEqual([
      { start_time: Date.parse(first.request_time), duration: 5000, tokens: 30, cost: 1, state: 'OK', assessments: {} },
    ]);
  });

  it('keeps missing duration absent while summing available metadata and retaining empty run slots', () => {
    const traces = [
      makeTrace('known', {
        execution_duration: '2s',
        trace_metadata: {
          'mlflow.trace.tokenUsage': JSON.stringify({ total_tokens: 0 }),
          'mlflow.trace.cost': JSON.stringify({ total_cost: 0 }),
        },
      }),
      makeTrace('missing', { execution_duration: undefined, trace_metadata: {} }),
    ];
    const summaries = getSessionComparisonSummaries(makeRows([[traces, []]]), []).get('row-0');
    expect(summaries).toEqual([
      {
        start_time: Date.parse(traces[0].request_time),
        duration: undefined,
        tokens: 0,
        cost: 0,
        state: 'OK',
        assessments: {},
      },
      {
        start_time: undefined,
        duration: undefined,
        tokens: undefined,
        cost: undefined,
        state: undefined,
        assessments: {},
      },
    ]);
  });
});

// @ui-test-skill-generated
describe('getEvalRunSessionComparisonColumnAggregates', () => {
  it.each([
    { dtype: 'boolean', value: true, displayValue: '100%' },
    { dtype: 'numeric', value: 0.1, displayValue: '0.1' },
  ] satisfies { dtype: SessionComparisonAssessmentInfo['dtype']; value: number | boolean; displayValue: string }[])(
    'does not report $dtype score changes when equivalent sessions have different turn counts',
    ({ dtype, value, displayValue }) => {
      const rows = makeRows([
        [
          [scoredTrace('baseline', value)],
          Array.from({ length: 6 }, (_, index) => scoredTrace(`candidate-${index}`, value)),
        ],
      ]);
      const infos: SessionComparisonAssessmentInfo[] = [{ ...NUMERIC_INFO, dtype }];
      const aggregate = getEvalRunSessionComparisonColumnAggregates(
        rows,
        getSessionComparisonSummaries(rows, infos),
        infos,
        INTL,
        2,
      )['assessment:quality'];
      expect(aggregate.values).toEqual([
        { displayValue, displayDelta: undefined, changeCounts: undefined },
        { displayValue, displayDelta: undefined, delta: 0, percentDelta: 0, changeCounts: undefined },
      ]);
    },
  );

  it('weights session means and standalone traces equally and counts only matched-session changes', () => {
    const rows: TracesComparisonRow[] = [
      ...makeRows([
        [[scoredTrace('base-1', 0), scoredTrace('base-2', 4)], [scoredTrace('candidate-1', 6)]],
        [[scoredTrace('base-3', 10)], []],
      ]),
      {
        id: 'standalone',
        input: 'Standalone input',
        runs: [
          { runUuid: 'run-0', label: 'Run 0', color: 'blue', trace: scoredTrace('base-standalone', 0) },
          { runUuid: 'run-1', label: 'Run 1', color: 'green', trace: scoredTrace('candidate-standalone', -2) },
        ],
      },
    ];
    const infos = [NUMERIC_INFO];
    const summaries = getSessionComparisonSummaries(rows, infos);
    expect(getEvalRunSessionComparisonColumnAggregates(rows, summaries, infos, INTL, 2)['assessment:quality']).toEqual({
      label: 'AVG',
      values: [
        { displayValue: '4', displayDelta: undefined, changeCounts: undefined },
        {
          displayValue: '2',
          displayDelta: '-2',
          delta: -2,
          percentDelta: -50,
          changeCounts: { improved: 1, regressed: 1 },
        },
      ],
    });
  });

  it('averages per-session boolean rates and reports completed-session failure rate with lower-is-better counts', () => {
    const rows = makeRows([
      [
        [scoredTrace('base-1', false), { ...scoredTrace('base-2', true), state: 'ERROR' }],
        [scoredTrace('candidate-1', true)],
      ],
      [[scoredTrace('base-3', false)], [{ ...scoredTrace('candidate-2', false), state: 'ERROR' }]],
      [[{ ...scoredTrace('pending', true), state: 'IN_PROGRESS' }], []],
    ]);
    const infos: SessionComparisonAssessmentInfo[] = [{ ...NUMERIC_INFO, dtype: 'boolean' }];
    const aggregates = getEvalRunSessionComparisonColumnAggregates(
      rows,
      getSessionComparisonSummaries(rows, infos),
      infos,
      INTL,
      2,
    );
    expect(aggregates['assessment:quality']).toEqual({
      label: 'TRUE',
      values: [
        { displayValue: '50%', displayDelta: undefined, changeCounts: undefined },
        {
          displayValue: '50%',
          displayDelta: undefined,
          delta: 0,
          percentDelta: 0,
          changeCounts: { improved: 1, regressed: 0 },
        },
      ],
    });
    expect(aggregates['state']).toEqual({
      label: 'ERROR RATE',
      values: [
        { displayValue: '50%', displayDelta: undefined, changeCounts: undefined },
        {
          displayValue: '50%',
          displayDelta: undefined,
          delta: 0,
          percentDelta: 0,
          changeCounts: { improved: 1, regressed: 1 },
        },
      ],
    });
  });

  it('keeps categorical summaries without numeric deltas', () => {
    const rows = makeRows([
      [[scoredTrace('a', 'good'), scoredTrace('b', 'good')], [scoredTrace('c', 'fair')]],
      [[scoredTrace('d', 'bad')], []],
    ]);
    const infos: SessionComparisonAssessmentInfo[] = [{ ...NUMERIC_INFO, dtype: 'string' }];
    expect(
      getEvalRunSessionComparisonColumnAggregates(rows, getSessionComparisonSummaries(rows, infos), infos, INTL, 2)[
        'assessment:quality'
      ],
    ).toEqual({ values: [{ displayValue: '2 unique' }, { displayValue: '1 unique' }] });
  });
});

// @ui-test-skill-generated
describe('sortSessionComparisonRows', () => {
  it('sorts recorded categorical session feedback by value rather than by distinct count', () => {
    const rows = makeRows([
      [[makeTrace('z', { assessments: [sessionScore('zebra')] })]],
      [[makeTrace('a', { assessments: [sessionScore('ant')] })]],
    ]);
    const infos: SessionComparisonAssessmentInfo[] = [{ name: 'quality', scope: 'session', dtype: 'string' }];
    expect(
      sortSessionComparisonRows(
        rows,
        getSessionComparisonSummaries(rows, infos),
        'assessment:quality',
        'asc',
        infos,
      ).map((row) => row.id),
    ).toEqual(['row-1', 'row-0']);
  });

  it('sorts categorical trace summaries by distinct-value count', () => {
    const rows = makeRows([
      [[scoredTrace('a', 'alpha'), scoredTrace('b', 'beta')]],
      [[scoredTrace('c', 'gamma'), scoredTrace('d', 'gamma')]],
    ]);
    const infos: SessionComparisonAssessmentInfo[] = [{ name: 'quality', scope: 'trace', dtype: 'string' }];
    expect(
      sortSessionComparisonRows(
        rows,
        getSessionComparisonSummaries(rows, infos),
        'assessment:quality',
        'asc',
        infos,
      ).map((row) => row.id),
    ).toEqual(['row-1', 'row-0']);
  });

  it.each([
    { dir: 'asc', expected: ['row-1', 'row-0', 'row-2', 'row-3'] },
    { dir: 'desc', expected: ['row-0', 'row-1', 'row-2', 'row-3'] },
  ] satisfies { dir: 'asc' | 'desc'; expected: string[] }[])(
    'sorts summaries baseline-first with run tie-breaks and missing values last in $dir order',
    ({ dir, expected }) => {
      const rows = makeRows([
        [[scoredTrace('base-high', 2)], [scoredTrace('candidate-high', 5)]],
        [[scoredTrace('base-low', 2)], [scoredTrace('candidate-low', 1)]],
        [[], [scoredTrace('candidate-only', 100)]],
        [[], []],
      ]);
      const summaries = getSessionComparisonSummaries(rows, [NUMERIC_INFO]);
      expect(sortSessionComparisonRows(rows, summaries, 'assessment:quality', dir).map((row) => row.id)).toEqual(
        expected,
      );
      expect(rows.map((row) => row.id)).toEqual(['row-0', 'row-1', 'row-2', 'row-3']);
      expect(rows.flatMap((row) => row.runs.flatMap((run) => run.traces.map((trace) => trace.trace_id)))).toEqual([
        'base-high',
        'candidate-high',
        'base-low',
        'candidate-low',
        'candidate-only',
      ]);
    },
  );
});
