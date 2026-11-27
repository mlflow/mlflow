import { describe, expect, it } from '@jest/globals';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import type { TraceComparisonRow, TraceSortColumnId } from '@databricks/web-shared/traces-table';
import {
  makeFeedbackAssessment,
  makeTrace,
} from '../../components/experiment-page/components/traces-v4/test-utils/mockTraces';
import { sortEvalRunComparisonRows } from './evalRunComparisonTraceSort';

const row = (id: string, ...traces: (ModelTraceInfoV3 | undefined)[]): TraceComparisonRow => ({
  id,
  input: id,
  runs: traces.map((trace, index) => ({ trace, runUuid: `run-${index}`, label: `Run ${index}`, color: 'blue' })),
});
const sortCases: { column: TraceSortColumnId; low: Partial<ModelTraceInfoV3>; high: Partial<ModelTraceInfoV3> }[] = [
  { column: 'duration', low: { execution_duration: '2s' }, high: { execution_duration: '10s' } },
  { column: 'start_time', low: { request_time: '2025-01-01' }, high: { request_time: '2026-01-01' } },
  {
    column: 'tokens',
    low: { trace_metadata: { 'mlflow.trace.tokenUsage': '{"total_tokens":2}' } },
    high: { trace_metadata: { 'mlflow.trace.tokenUsage': '{"total_tokens":10}' } },
  },
  {
    column: 'cost',
    low: { trace_metadata: { 'mlflow.trace.cost': '{"total_cost":0.02}' } },
    high: { trace_metadata: { 'mlflow.trace.cost': '{"total_cost":0.1}' } },
  },
  { column: 'state', low: { state: 'ERROR' }, high: { state: 'OK' } },
];

describe('sortEvalRunComparisonRows', () => {
  it.each(sortCases)('sorts raw $column baseline values with candidates breaking ties', ({ column, low, high }) => {
    const a = row('a', makeTrace('a', low), makeTrace('a-candidate', high));
    const b = row('b', makeTrace('b', low), makeTrace('b-candidate', low));
    const c = row('c', makeTrace('c', high));
    const rows = [c, a, b];
    expect(sortEvalRunComparisonRows(rows, column, 'asc')).toEqual([b, a, c]);
    expect(sortEvalRunComparisonRows(rows, column, 'desc')).toEqual([c, a, b]);
    expect(rows).toEqual([c, a, b]);
    expect(a.runs.map(({ trace }) => trace?.trace_id)).toEqual(['a', 'a-candidate']);
  });

  it.each(['asc', 'desc'] as const)('puts absent/invalid/error values last for %s', (direction) => {
    const present = row('present', makeTrace('p', { assessments: [makeFeedbackAssessment('q', 1)] }));
    const missing = row('missing', undefined, makeTrace('m', { assessments: [makeFeedbackAssessment('q', 100)] }));
    const failed = row(
      'failed',
      makeTrace('f', {
        assessments: [makeFeedbackAssessment('q', 200, { error: { error_code: 'FAILED', error_message: 'failed' } })],
      }),
    );
    expect(sortEvalRunComparisonRows([failed, missing, present], 'assessment:q', direction)).toEqual([
      present,
      missing,
      failed,
    ]);
    expect(
      sortEvalRunComparisonRows(
        [row('bad', makeTrace('bad', { request_time: 'invalid' })), row('valid', makeTrace('valid'))],
        'start_time',
        direction,
      ).map(({ id }) => id),
    ).toEqual(['valid', 'bad']);
  });

  it('uses displayed latest assessments, deterministic mixed types and stable group identities', () => {
    const numeric = row(
      'n',
      makeTrace('n', {
        assessments: [makeFeedbackAssessment('q', 100), makeFeedbackAssessment('q', 2, { create_time: '2026-01-01' })],
      }),
    );
    const no = row('no', makeTrace('no', { assessments: [makeFeedbackAssessment('q', false)] }));
    const yes = row('yes', makeTrace('yes', { assessments: [makeFeedbackAssessment('q', true)] }));
    const text = row('s', makeTrace('s', { assessments: [makeFeedbackAssessment('q', 'a')] }));
    expect(sortEvalRunComparisonRows([text, yes, numeric, no], 'assessment:q', 'asc')).toEqual([
      numeric,
      no,
      yes,
      text,
    ]);
    expect(
      sortEvalRunComparisonRows([row('z', makeTrace('z')), row('a', makeTrace('a'))], 'duration', 'desc').map(
        ({ id }) => id,
      ),
    ).toEqual(['a', 'z']);
    expect(sortEvalRunComparisonRows([], 'start_time', 'desc')).toEqual([]);
    expect(sortEvalRunComparisonRows([numeric], 'start_time', 'desc')).toEqual([numeric]);
  });
});
