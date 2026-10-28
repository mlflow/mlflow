import { describe, expect, test } from '@jest/globals';
import { createIntl } from 'react-intl';
import {
  buildEvalRunComparisonSnapshot,
  normalizeComparisonRunUuids,
  type EvalComparisonRunWithTraces,
} from './evalRunComparisonData';
import { createComparisonTrace } from './evalRunComparisonTestUtils';

const intl = createIntl({ locale: 'en' });
const snapshotFor = (runs: EvalComparisonRunWithTraces[]) =>
  buildEvalRunComparisonSnapshot(intl, runs, new Map(), new Set());
const baseRuns = [
  { runUuid: 'baseline', traceInfos: [createComparisonTrace('baseline', { quality: true, latency: 10 })] },
  { runUuid: 'candidate', traceInfos: [createComparisonTrace('candidate', { quality: false, latency: 5 })] },
];

describe('comparison facts', () => {
  test('normalizes duplicate selections while preserving the baseline and order', () => {
    expect(normalizeComparisonRunUuids('a', ['b', 'a', '', 'b', 'c'])).toEqual(['a', 'b', 'c']);
  });

  test('computes passing-rate deltas and treats numeric changes neutrally', () => {
    const snapshot = snapshotFor(baseRuns);
    expect(snapshot.deltas).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: 'quality',
          baselineValue: 100,
          value: 0,
          delta: -100,
          direction: 'regressed',
          unit: 'percentagePoints',
        }),
        expect.objectContaining({
          name: 'latency',
          baselineValue: 10,
          value: 5,
          delta: -5,
          direction: 'neutral',
          unit: 'raw',
        }),
      ]),
    );
    expect(snapshot.comparisons).toEqual([
      { runUuid: 'candidate', improved: 0, regressed: 1, unchanged: 0, numericChanges: 1 },
    ]);
    expect(snapshot.leaderRunUuid).toBe('baseline');
  });

  test('compares yes/no and boolean success rates using the same scale', () => {
    const snapshot = snapshotFor([
      {
        runUuid: 'a',
        traceInfos: [createComparisonTrace('a', { quality: 'yes' }), createComparisonTrace('a', { quality: 'no' }, 1)],
      },
      { runUuid: 'b', traceInfos: [createComparisonTrace('b', { quality: true })] },
    ]);
    expect(snapshot.deltas).toEqual([
      expect.objectContaining({ baselineValue: 50, value: 100, delta: 50, direction: 'improved' }),
    ]);
  });

  test('does not substitute zero for missing or errored scores', () => {
    const trace = createComparisonTrace('b', { quality: false });
    const assessment = trace.assessments?.[0];
    if (assessment) {
      assessment.error = { error_code: 'FAILED', error_message: 'scorer failed' };
    }
    const snapshot = snapshotFor([
      baseRuns[0],
      { runUuid: 'b', traceInfos: [trace] },
      { runUuid: 'empty', traceInfos: [] },
    ]);
    expect(snapshot.deltas).toEqual([]);
    expect(snapshot.commonSuccessScorerCount).toBe(0);
    expect(snapshot.leaderRunUuid).toBeUndefined();
  });

  test('does not compare errored numeric scores', () => {
    const trace = createComparisonTrace('b', { latency: 5 });
    const assessment = trace.assessments?.[0];
    if (assessment) {
      assessment.error = { error_code: 'FAILED', error_message: 'scorer failed' };
    }
    const snapshot = snapshotFor([
      { runUuid: 'a', traceInfos: [createComparisonTrace('a', { latency: 10 })] },
      { runUuid: 'b', traceInfos: [trace] },
    ]);
    expect(snapshot.deltas).toEqual([]);
  });

  test('does not compare incompatible scorer types or categorical results', () => {
    const snapshot = snapshotFor([
      baseRuns[0],
      { runUuid: 'b', traceInfos: [createComparisonTrace('b', { quality: 0.5, category: 'excellent' })] },
    ]);
    expect(snapshot.deltas).toEqual([]);
    expect(snapshot.leaderRunUuid).toBeUndefined();
  });

  test('ranks only common success scorers, weighting scorers equally rather than by trace count', () => {
    const snapshot = snapshotFor([
      {
        runUuid: 'a',
        traceInfos: [
          createComparisonTrace('a', { quality: true, safety: 'no', extra: true }),
          createComparisonTrace('a', { quality: true }, 1),
        ],
      },
      { runUuid: 'b', traceInfos: [createComparisonTrace('b', { quality: false, safety: 'yes' })] },
      { runUuid: 'c', traceInfos: [createComparisonTrace('c', { quality: true, safety: 'yes' })] },
    ]);
    expect(snapshot.commonSuccessScorerCount).toBe(2);
    expect(snapshot.leaderRunUuid).toBe('c');
    expect(
      snapshotFor([
        {
          runUuid: 'a',
          traceInfos: [
            createComparisonTrace('a', { quality: true, safety: 'no' }),
            createComparisonTrace('a', { quality: true }, 1),
          ],
        },
        { runUuid: 'b', traceInfos: [createComparisonTrace('b', { quality: false, safety: 'yes' })] },
      ]).leaderRunUuid,
    ).toBeUndefined();
  });

  test('keeps truncation and run-name fallback in the snapshot', () => {
    const snapshot = buildEvalRunComparisonSnapshot(
      intl,
      baseRuns,
      new Map([['baseline', 'Baseline name']]),
      new Set(['candidate']),
    );
    expect(snapshot.runs).toEqual([
      expect.objectContaining({ runName: 'Baseline name', isTruncated: false }),
      expect.objectContaining({ runName: 'candidate', isTruncated: true }),
    ]);
    expect(snapshot.leaderRunUuid).toBeUndefined();
  });
});
