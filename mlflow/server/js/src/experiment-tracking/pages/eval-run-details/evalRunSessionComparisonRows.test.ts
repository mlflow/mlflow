import { describe, expect, it } from '@jest/globals';
import { SESSION_ID_METADATA_KEY, type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { makeTrace } from '../../components/experiment-page/components/traces-v4/test-utils/mockTraces';
import type { EvalComparisonRunWithTraces } from './evalRunComparisonData';
import { buildEvalRunSessionComparisonRows } from './evalRunSessionComparisonRows';

const RUN_NAMES = new Map([
  ['baseline', 'Baseline'],
  ['candidate', 'Candidate'],
]);
const getRunColor = (runUuid: string) => `color-${runUuid}`;
const scenario = { 'mlflow.simulation.goal': 'Book a flight', 'mlflow.simulation.persona': 'Traveler' };
const sessionTrace = (
  traceId: string,
  sessionId: string,
  overrides: Partial<ModelTraceInfoV3> = {},
): ModelTraceInfoV3 =>
  makeTrace(traceId, {
    ...overrides,
    trace_metadata: { [SESSION_ID_METADATA_KEY]: sessionId, ...overrides.trace_metadata },
  });
const sessionIds = (rows: ReturnType<typeof buildEvalRunSessionComparisonRows>) =>
  rows.map((row) => (row.kind === 'session' ? row.runs.map((run) => run.sessionId) : undefined));

// @ui-test-skill-generated
describe('buildEvalRunSessionComparisonRows', () => {
  it('matches complete sessions by scenario and keeps each run’s turns chronological', () => {
    const baselineFirst = sessionTrace('baseline-first', 'baseline-session', {
      trace_metadata: scenario,
      request_time: '2026-01-01T00:00:00Z',
      request_preview: 'Different baseline input',
    });
    const baselineLast = sessionTrace('baseline-last', 'baseline-session', {
      trace_metadata: scenario,
      request_time: '2026-01-01T00:01:00Z',
    });
    const candidate = sessionTrace('candidate-only', 'candidate-session', { trace_metadata: scenario });
    const rows = buildEvalRunSessionComparisonRows(
      [
        { runUuid: 'baseline', traceInfos: [baselineLast, baselineFirst] },
        { runUuid: 'candidate', traceInfos: [candidate] },
        { runUuid: 'third', traceInfos: [] },
      ],
      RUN_NAMES,
      getRunColor,
    );

    expect(rows).toHaveLength(1);
    const [row] = rows;
    expect(row.kind).toBe('session');
    if (row.kind !== 'session') {
      throw new Error('Expected a session comparison row');
    }
    expect(row.runs).toEqual([
      {
        runUuid: 'baseline',
        label: 'Baseline',
        color: 'color-baseline',
        sessionId: 'baseline-session',
        traces: [baselineFirst, baselineLast],
      },
      {
        runUuid: 'candidate',
        label: 'Candidate',
        color: 'color-candidate',
        sessionId: 'candidate-session',
        traces: [candidate],
      },
      { runUuid: 'third', label: 'third', color: 'color-third', sessionId: undefined, traces: [] },
    ]);
  });

  it.each<{ label: string; metadata: Record<string, string> }>([
    { label: 'missing', metadata: {} },
    { label: 'goal only', metadata: { 'mlflow.simulation.goal': 'Book a flight' } },
    { label: 'persona only', metadata: { 'mlflow.simulation.persona': 'Traveler' } },
  ])('falls back to actual session IDs for $label scenario metadata', ({ metadata }) => {
    const rows = buildEvalRunSessionComparisonRows(
      [
        { runUuid: 'baseline', traceInfos: [sessionTrace('base', 'shared-session', { trace_metadata: metadata })] },
        {
          runUuid: 'candidate',
          traceInfos: [
            sessionTrace('matched', 'shared-session', { trace_metadata: metadata }),
            sessionTrace('unmatched', 'other-session', { trace_metadata: metadata }),
          ],
        },
      ],
      RUN_NAMES,
      getRunColor,
    );
    expect(sessionIds(rows)).toEqual([
      ['shared-session', 'shared-session'],
      [undefined, 'other-session'],
    ]);
  });

  it.each<{ label: string; turns: Record<string, string>[] }>([
    { label: 'disagree about their scenario', turns: [scenario, { ...scenario, 'mlflow.simulation.goal': 'Cancel' }] },
    { label: 'are only partly annotated', turns: [scenario, {}] },
    {
      label: 'split the goal and persona',
      turns: [{ 'mlflow.simulation.goal': 'Book a flight' }, { 'mlflow.simulation.persona': 'Traveler' }],
    },
  ])('uses the session ID when turns $label', ({ turns }) => {
    const rows = buildEvalRunSessionComparisonRows(
      [
        {
          runUuid: 'baseline',
          traceInfos: turns.map((metadata, index) =>
            sessionTrace(`base-${index}`, 'shared-session', { trace_metadata: metadata }),
          ),
        },
        {
          runUuid: 'candidate',
          traceInfos: [
            sessionTrace('candidate', 'shared-session'),
            sessionTrace('scenario', 'other', { trace_metadata: scenario }),
          ],
        },
      ],
      RUN_NAMES,
      getRunColor,
    );
    expect(sessionIds(rows)).toEqual([
      ['shared-session', 'shared-session'],
      [undefined, 'other'],
    ]);
  });

  it('pairs repeated scenario sessions by their start time and session ID without merging them', () => {
    const runs: EvalComparisonRunWithTraces[] = [
      {
        runUuid: 'baseline',
        traceInfos: [
          sessionTrace('base-old-late-turn', 'base-old', {
            trace_metadata: scenario,
            request_time: '2026-01-03T00:00:00Z',
          }),
          sessionTrace('base-new-b', 'base-new-b', { trace_metadata: scenario, request_time: '2026-01-02T00:00:00Z' }),
          sessionTrace('base-old-start', 'base-old', {
            trace_metadata: scenario,
            request_time: '2026-01-01T00:00:00Z',
          }),
          sessionTrace('base-new-a', 'base-new-a', { trace_metadata: scenario, request_time: '2026-01-02T00:00:00Z' }),
        ],
      },
      {
        runUuid: 'candidate',
        traceInfos: [
          sessionTrace('candidate-old', 'candidate-old', {
            trace_metadata: scenario,
            request_time: '2026-01-01T00:00:00Z',
          }),
          sessionTrace('candidate-new', 'candidate-new', {
            trace_metadata: scenario,
            request_time: '2026-01-02T00:00:00Z',
          }),
        ],
      },
    ];
    const rows = buildEvalRunSessionComparisonRows(runs, RUN_NAMES, getRunColor);
    expect(sessionIds(rows)).toEqual([
      ['base-new-a', 'candidate-new'],
      ['base-new-b', 'candidate-old'],
      ['base-old', undefined],
    ]);
    const reordered = buildEvalRunSessionComparisonRows(
      runs.map((run) => ({ ...run, traceInfos: [...run.traceInfos].reverse() })),
      RUN_NAMES,
      getRunColor,
    );
    expect(reordered).toEqual(rows);
  });

  it('orders invalid timestamps last and equal-time turns by trace ID without mutating inputs', () => {
    const traces = [
      sessionTrace('invalid-z', 'session', { request_time: 'invalid' }),
      sessionTrace('same-time-b', 'session'),
      sessionTrace('invalid-a', 'session', { request_time: '' }),
      sessionTrace('same-time-a', 'session'),
    ];
    const original = [...traces];
    const [row] = buildEvalRunSessionComparisonRows(
      [{ runUuid: 'baseline', traceInfos: traces }],
      RUN_NAMES,
      getRunColor,
    );
    expect(row.kind).toBe('session');
    if (row.kind !== 'session') {
      throw new Error('Expected a session comparison row');
    }
    expect(row.runs.flatMap((run) => run.traces.map((trace) => trace.trace_id))).toEqual([
      'same-time-a',
      'same-time-b',
      'invalid-a',
      'invalid-z',
    ]);
    expect(traces).toEqual(original);
  });

  it('does not collide delimiter-containing scenario pairs or session IDs with scenario keys', () => {
    const rows = buildEvalRunSessionComparisonRows(
      [
        {
          runUuid: 'baseline',
          traceInfos: [
            sessionTrace('base', 'metadata:a:b:c', {
              trace_metadata: { 'mlflow.simulation.goal': 'a:b', 'mlflow.simulation.persona': 'c' },
            }),
          ],
        },
        {
          runUuid: 'candidate',
          traceInfos: [
            sessionTrace('candidate', 'another', {
              trace_metadata: { 'mlflow.simulation.goal': 'a', 'mlflow.simulation.persona': 'b:c' },
            }),
            sessionTrace('session-only', 'metadata:a:b:c'),
          ],
        },
      ],
      RUN_NAMES,
      getRunColor,
    );
    expect(sessionIds(rows)).toEqual([
      ['metadata:a:b:c', undefined],
      [undefined, 'another'],
      [undefined, 'metadata:a:b:c'],
    ]);
  });

  it('keeps sessionless traces in full-input comparison even when scenario metadata exists', () => {
    const rows = buildEvalRunSessionComparisonRows(
      [
        {
          runUuid: 'baseline',
          traceInfos: [makeTrace('base', { request_preview: 'Shared prefix', trace_metadata: scenario })],
        },
        {
          runUuid: 'candidate',
          traceInfos: [makeTrace('candidate', { request_preview: 'Shared prefix', trace_metadata: scenario })],
        },
      ],
      RUN_NAMES,
      getRunColor,
      new Map([
        ['base', 'Full input A'],
        ['candidate', 'Full input B'],
      ]),
    );
    expect(
      rows.map((row) => (row.kind === 'session' ? 'session' : row.runs.map((run) => run.trace?.trace_id))),
    ).toEqual([
      ['base', undefined],
      [undefined, 'candidate'],
    ]);
  });

  it('returns no comparison rows when all runs are empty', () => {
    expect(
      buildEvalRunSessionComparisonRows(
        [
          { runUuid: 'baseline', traceInfos: [] },
          { runUuid: 'candidate', traceInfos: [] },
        ],
        RUN_NAMES,
        getRunColor,
      ),
    ).toEqual([]);
  });
});
