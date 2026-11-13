import { describe, expect, it } from '@jest/globals';
import {
  makeFeedbackAssessment,
  makeTrace,
} from '../../components/experiment-page/components/traces-v4/test-utils/mockTraces';
import type { EvalComparisonRunWithTraces } from './evalRunComparisonData';
import { buildEvalRunTraceComparisonRows } from './evalRunTraceComparisonRows';

describe('buildEvalRunTraceComparisonRows', () => {
  const baseline = makeTrace('baseline-id', { request_preview: 'Shared input', response_preview: 'Baseline answer' });
  const candidate = makeTrace('candidate-id', {
    request_preview: 'Shared input',
    response_preview: 'Candidate answer',
  });
  const unmatched = makeTrace('unmatched-id', { request_preview: 'Unmatched input' });
  const runs: EvalComparisonRunWithTraces[] = [
    { runUuid: 'baseline', traceInfos: [baseline] },
    { runUuid: 'candidate', traceInfos: [candidate, unmatched] },
    { runUuid: 'third', traceInfos: [] },
  ];
  const names = new Map([
    ['baseline', 'Baseline'],
    ['candidate', 'Candidate'],
  ]);
  const color = (id: string) => `color-${id}`;

  it.each(['recorded input', ''])('prefers recorded span input %j while displaying the preview', (input) => {
    const rows = buildEvalRunTraceComparisonRows(
      [
        { runUuid: 'baseline', traceInfos: [{ ...baseline, request: 'shared fallback' }] },
        { runUuid: 'candidate', traceInfos: [{ ...candidate, request: 'shared fallback' }] },
      ],
      names,
      color,
      '',
      new Map([
        ['baseline-id', input],
        ['candidate-id', 'different recorded input'],
      ]),
    );
    expect(rows.map((row) => ({ input: row.input, traces: row.runs.map((run) => run.trace?.trace_id) }))).toEqual([
      { input: 'Shared input', traces: ['baseline-id', undefined] },
      { input: 'Shared input', traces: [undefined, 'candidate-id'] },
    ]);
  });

  it('keeps distinct full requests separate despite identical previews and input metadata', () => {
    const preview = 'x'.repeat(10 * 1024);
    const rows = buildEvalRunTraceComparisonRows(
      [
        {
          runUuid: 'baseline',
          traceInfos: [
            makeTrace('baseline-id', {
              request: `${preview}baseline suffix`,
              request_preview: preview,
              trace_metadata: { 'mlflow.traceInputs': 'Shared metadata input' },
            }),
          ],
        },
        {
          runUuid: 'candidate',
          traceInfos: [
            makeTrace('candidate-id', {
              request: `${preview}candidate suffix`,
              request_preview: preview,
              trace_metadata: { 'mlflow.traceInputs': 'Shared metadata input' },
            }),
          ],
        },
      ],
      names,
      color,
      '',
    );
    expect(rows.map((row) => row.runs.map((run) => run.trace?.trace_id))).toEqual([
      ['baseline-id', undefined],
      [undefined, 'candidate-id'],
    ]);
  });

  it('uses full input metadata before the preview when the request field is absent', () => {
    const rows = buildEvalRunTraceComparisonRows(
      [
        {
          runUuid: 'baseline',
          traceInfos: [{ ...baseline, trace_metadata: { 'mlflow.traceInputs': 'Shared input with baseline suffix' } }],
        },
        {
          runUuid: 'candidate',
          traceInfos: [
            { ...candidate, trace_metadata: { 'mlflow.traceInputs': 'Shared input with candidate suffix' } },
          ],
        },
      ],
      names,
      color,
      '',
    );
    expect(rows.map((row) => row.runs.map((run) => run.trace?.trace_id))).toEqual([
      ['baseline-id', undefined],
      [undefined, 'candidate-id'],
    ]);
  });

  it.each([
    { label: 'nonempty', fullInput: '{"question":"Complete shared input"}' },
    { label: 'empty', fullInput: '' },
  ])('pairs duplicate $label full inputs across sources while still displaying previews', ({ fullInput }) => {
    const rows = buildEvalRunTraceComparisonRows(
      [
        {
          runUuid: 'baseline',
          traceInfos: ['baseline-1', 'baseline-2'].map((id) =>
            makeTrace(id, {
              request: fullInput,
              request_preview: 'Baseline preview',
              trace_metadata: { 'mlflow.traceInputs': 'Ignored metadata input' },
            }),
          ),
        },
        {
          runUuid: 'candidate',
          traceInfos: ['candidate-1', 'candidate-2'].map((id) =>
            makeTrace(id, {
              request_preview: 'Candidate preview',
              trace_metadata: { 'mlflow.traceInputs': fullInput },
            }),
          ),
        },
      ],
      names,
      color,
      '',
    );
    expect(rows.map((row) => ({ input: row.input, traceIds: row.runs.map((run) => run.trace?.trace_id) }))).toEqual([
      { input: 'Baseline preview', traceIds: ['baseline-1', 'candidate-1'] },
      { input: 'Baseline preview', traceIds: ['baseline-2', 'candidate-2'] },
    ]);
  });

  it('preserves ordered run slots, labels, colors and unmatched rows', () => {
    expect(buildEvalRunTraceComparisonRows(runs, names, color, '')).toEqual([
      {
        id: JSON.stringify(['baseline-id', 'candidate-id', null]),
        input: 'Shared input',
        runs: [
          { runUuid: 'baseline', label: 'Baseline', color: 'color-baseline', trace: baseline },
          { runUuid: 'candidate', label: 'Candidate', color: 'color-candidate', trace: candidate },
          { runUuid: 'third', label: 'third', color: 'color-third', trace: undefined },
        ],
      },
      {
        id: JSON.stringify([null, 'unmatched-id', null]),
        input: 'Unmatched input',
        runs: [
          { runUuid: 'baseline', label: 'Baseline', color: 'color-baseline', trace: undefined },
          { runUuid: 'candidate', label: 'Candidate', color: 'color-candidate', trace: unmatched },
          { runUuid: 'third', label: 'third', color: 'color-third', trace: undefined },
        ],
      },
    ]);
  });

  it.each(['SHARED INPUT', 'candidate-id', 'trace:/cat.sch/candidate-id'])(
    'retains nonmatching sibling traces when searching %s',
    (search) => {
      const rows = buildEvalRunTraceComparisonRows(runs, names, color, search);
      expect(rows.map((row) => row.runs.map((run) => run.trace?.trace_id))).toEqual([
        ['baseline-id', 'candidate-id', undefined],
      ]);
    },
  );

  it.each([
    { inputRuns: [], search: '' },
    { inputRuns: runs, search: 'not found' },
  ])('returns no rows for empty results ($search)', ({ inputRuns, search }) => {
    expect(buildEvalRunTraceComparisonRows(inputRuns, names, color, search)).toEqual([]);
  });

  it('keeps separate assessment values when two runs share a physical trace', () => {
    const first = makeTrace('shared-id', { assessments: [makeFeedbackAssessment('Quality', true)] });
    const second = { ...first, assessments: [makeFeedbackAssessment('Quality', false)] };
    const rows = buildEvalRunTraceComparisonRows(
      [
        { runUuid: 'baseline', traceInfos: [first] },
        { runUuid: 'candidate', traceInfos: [second] },
      ],
      names,
      color,
      '',
    );
    expect(rows.map((row) => row.runs.map((run) => run.trace?.assessments))).toEqual([
      [first.assessments, second.assessments],
    ]);
  });

  it.each([
    {
      counts: [2, 2, 2],
      expected: [
        ['baseline-0', 'candidate-0', 'third-0'],
        ['baseline-1', 'candidate-1', 'third-1'],
      ],
    },
    {
      counts: [3, 1, 2],
      expected: [
        ['baseline-0', 'candidate-0', 'third-0'],
        ['baseline-1', undefined, 'third-1'],
        ['baseline-2', undefined, undefined],
      ],
    },
    {
      counts: [0, 2, 3],
      expected: [
        [undefined, 'candidate-0', 'third-0'],
        [undefined, 'candidate-1', 'third-1'],
        [undefined, undefined, 'third-2'],
      ],
    },
  ])('aligns duplicate occurrences without dropping traces for counts $counts', ({ counts, expected }) => {
    const rows = buildEvalRunTraceComparisonRows(
      ['baseline', 'candidate', 'third'].map((runUuid, runIndex) => ({
        runUuid,
        traceInfos: Array.from({ length: counts[runIndex] }, (_, index) =>
          makeTrace(`${runUuid}-${index}`, { request_preview: 'Shared input' }),
        ),
      })),
      names,
      color,
      '',
    );
    expect(rows.map((row) => row.runs.map((run) => run.trace?.trace_id))).toEqual(expected);
  });

  it('pairs duplicates newest-first with trace-ID tie-breaking without mutating the inputs', () => {
    const inputRuns = ['baseline', 'candidate'].map((runUuid) => ({
      runUuid,
      traceInfos: [
        makeTrace(`${runUuid}-a-old`, { request_preview: 'Shared input', request_time: '2025-01-01T00:00:00Z' }),
        makeTrace(`${runUuid}-c-tie`, { request_preview: 'Shared input', request_time: '2025-01-02T00:00:00Z' }),
        makeTrace(`${runUuid}-z-new`, { request_preview: 'Shared input', request_time: '2025-01-03T00:00:00Z' }),
        makeTrace(`${runUuid}-b-tie`, { request_preview: 'Shared input', request_time: '2025-01-02T00:00:00Z' }),
      ],
    }));
    const originalTraceArrays = inputRuns.map((run) => [...run.traceInfos]);
    const expected = [
      ['baseline-z-new', 'candidate-z-new'],
      ['baseline-b-tie', 'candidate-b-tie'],
      ['baseline-c-tie', 'candidate-c-tie'],
      ['baseline-a-old', 'candidate-a-old'],
    ];
    const rows = buildEvalRunTraceComparisonRows(inputRuns, names, color, '');
    expect(rows.map((row) => row.runs.map((run) => run.trace?.trace_id))).toEqual(expected);
    expect(inputRuns.map((run) => run.traceInfos)).toEqual(originalTraceArrays);

    const shuffledRows = buildEvalRunTraceComparisonRows(
      inputRuns.map((run) => ({ ...run, traceInfos: [...run.traceInfos].reverse() })),
      names,
      color,
      '',
    );
    expect(shuffledRows.map((row) => row.runs.map((run) => run.trace?.trace_id))).toEqual(expected);
  });

  it('keeps duplicate ordinals per input and preserves baseline-first input ordering', () => {
    const firstInputBaseline = makeTrace('z-first', { request_preview: 'First input' });
    const firstInputCandidate = makeTrace('z-candidate', { request_preview: 'First input' });
    const repeatedBaseline = makeTrace('baseline-repeat', { request_preview: 'Shared input' });
    const repeatedCandidate = makeTrace('candidate-repeat', { request_preview: 'Shared input' });
    const rows = buildEvalRunTraceComparisonRows(
      [
        { runUuid: 'baseline', traceInfos: [firstInputBaseline, baseline, unmatched, repeatedBaseline] },
        { runUuid: 'candidate', traceInfos: [candidate, firstInputCandidate, repeatedCandidate] },
      ],
      names,
      color,
      '',
    );
    expect(rows.map((row) => row.runs.map((run) => run.trace?.trace_id))).toEqual([
      ['z-first', 'z-candidate'],
      ['baseline-id', 'candidate-id'],
      ['baseline-repeat', 'candidate-repeat'],
      ['unmatched-id', undefined],
    ]);
  });

  it('retains the matched duplicate sibling when searching by a repeated trace ID', () => {
    const repeatedBaseline = makeTrace('baseline-repeat', { request_preview: 'Shared input' });
    const repeatedCandidate = makeTrace('candidate-repeat', { request_preview: 'Shared input' });
    const rows = buildEvalRunTraceComparisonRows(
      [
        { runUuid: 'baseline', traceInfos: [baseline, repeatedBaseline] },
        { runUuid: 'candidate', traceInfos: [candidate, repeatedCandidate] },
      ],
      names,
      color,
      'candidate-repeat',
    );
    expect(rows.map((row) => row.runs.map((run) => run.trace?.trace_id))).toEqual([
      ['baseline-repeat', 'candidate-repeat'],
    ]);
  });
});
