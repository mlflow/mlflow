import { describe, expect, test } from '@jest/globals';
import { createIntl } from 'react-intl';
import {
  buildEvalRunComparisonScorers,
  buildEvalRunComparisonScorerRows,
  type EvalRunComparisonScorerRun,
} from './evalRunComparisonScorerData';
import { createComparisonTrace } from './evalRunComparisonTestUtils';

const intl = createIntl({ locale: 'en' });
const createRun = (
  runUuid: string,
  scores: Parameters<typeof createComparisonTrace>[1],
): EvalRunComparisonScorerRun => ({
  runUuid,
  runName: runUuid,
  status: 'ready',
  isTruncated: false,
  scorers: buildEvalRunComparisonScorers(intl, [createComparisonTrace(runUuid, scores)]),
});

// Unit: trace inputs produce the same score formatting used by the single-run Overview.
// @ui-test-skill-generated
describe('buildEvalRunComparisonScorers', () => {
  test('formats numeric averages, success rates and categorical counts', () => {
    const scores = buildEvalRunComparisonScorers(intl, [
      createComparisonTrace('run', { numeric: 1.234, boolean: true, passFail: 'yes', category: 'good' }),
      createComparisonTrace('run', { numeric: 2, boolean: false, passFail: 'no', category: 'great' }, 1),
    ]);
    expect(
      Object.fromEntries(
        scores.map(({ assessmentInfo, displayScore, kind, value }) => [
          assessmentInfo.name,
          { displayScore, kind, value },
        ]),
      ),
    ).toEqual({
      numeric: { displayScore: '1.62', kind: 'numeric', value: 1.617 },
      boolean: { displayScore: '50%', kind: 'rate', value: 50 },
      passFail: { displayScore: '50%', kind: 'rate', value: 50 },
      category: { displayScore: '2 values', kind: 'categorical', value: undefined },
    });
    expect(scores.find(({ assessmentInfo }) => assessmentInfo.name === 'category')?.valueCounts).toEqual([
      { value: 'good', count: 1 },
      { value: 'great', count: 1 },
    ]);
  });

  test('preserves genuine zero values and omits trailing decimal zeros', () => {
    expect(
      createRun('run', { numeric: 0, rate: false, integer: 2 }).scorers.map(({ displayScore }) => displayScore),
    ).toEqual(expect.arrayContaining(['0', '0%', '2']));
  });

  test('keeps errored, missing and nonfinite results unavailable', () => {
    const trace = createComparisonTrace('run', { failed: true, missing: true, nonfinite: NaN });
    const failed = trace.assessments?.find(({ assessment_name }) => assessment_name === 'failed');
    const missing = trace.assessments?.find(({ assessment_name }) => assessment_name === 'missing');
    if (!failed || !missing || !('feedback' in missing)) throw new Error('Missing fixture assessments');
    failed.error = { error_code: 'FAILED', error_message: 'Scorer failed' };
    missing.feedback = { value: undefined };
    const scores = buildEvalRunComparisonScorers(intl, [trace]);
    expect(scores).toHaveLength(3);
    expect(scores.map(({ displayScore, value }) => ({ displayScore, value }))).toEqual([
      { displayScore: '—', value: undefined },
      { displayScore: '—', value: undefined },
      { displayScore: '—', value: undefined },
    ]);
  });

  test('summarizes a session-level scorer over the sessions, not every turn', () => {
    // Two sessions of two turns; the session judge sits on each session's first turn.
    const traces = [0, 1, 2, 3].map((index) => createComparisonTrace('run', { correctness: true }, index));
    for (const [index, value] of [
      [0, 'none'],
      [2, 'high'],
    ] as const) {
      traces[index].assessments?.push({
        ...traces[index].assessments[0],
        assessment_id: `session-${index}`,
        assessment_name: 'frustration',
        feedback: { value },
        metadata: { 'mlflow.trace.session': `session-${index}` },
      });
    }
    const frustration = buildEvalRunComparisonScorers(intl, traces).find(
      ({ assessmentInfo }) => assessmentInfo.name === 'frustration',
    );
    expect(frustration?.valueCounts).toEqual(
      expect.arrayContaining([
        { value: 'none', count: 1 },
        { value: 'high', count: 1 },
      ]),
    );
    expect(frustration?.valueCounts).toHaveLength(2);
  });

  test('counts only scored categories, not missing or errored results', () => {
    const errored = createComparisonTrace('run', { category: 'bad' }, 2);
    if (errored.assessments?.[0]) {
      errored.assessments[0].error = { error_code: 'FAILED', error_message: 'Scorer failed' };
    }
    const scores = buildEvalRunComparisonScorers(intl, [
      createComparisonTrace('run', { category: 'good' }),
      createComparisonTrace('run', { other: true }, 1),
      errored,
    ]);
    expect(scores.find(({ assessmentInfo }) => assessmentInfo.name === 'category')?.displayScore).toBe('1 value');
  });

  test('lists each scored numeric value for the breakdown tooltip', () => {
    const errored = createComparisonTrace('run', { rating: 1 }, 3);
    if (errored.assessments?.[0]) {
      errored.assessments[0].error = { error_code: 'FAILED', error_message: 'Scorer failed' };
    }
    const scores = buildEvalRunComparisonScorers(intl, [
      createComparisonTrace('run', { rating: 4 }),
      createComparisonTrace('run', { rating: 4 }, 1),
      createComparisonTrace('run', { rating: 2 }, 2),
      errored,
    ]);
    expect(scores.find(({ assessmentInfo }) => assessmentInfo.name === 'rating')?.valueCounts).toEqual([
      { value: 4, count: 2 },
      { value: 2, count: 1 },
    ]);
  });

  test('returns no scorers for an empty trace set', () => {
    expect(buildEvalRunComparisonScorers(intl, [])).toEqual([]);
  });
});

// Unit: row identities, comparisons and semantic ranking are pure derived data.
// @ui-test-skill-generated
describe('buildEvalRunComparisonScorerRows', () => {
  test('unions all scorers and computes compatible deltas against the first run', () => {
    const rows = buildEvalRunComparisonScorerRows([
      createRun('baseline', { quality: false, numeric: 2, baselineOnly: true }),
      createRun('candidate', { quality: true, numeric: 3.5, candidateOnly: 'good' }),
    ]);
    expect(rows.map(({ assessmentInfo }) => assessmentInfo.name)).toEqual([
      'baselineOnly',
      'candidateOnly',
      'numeric',
      'quality',
    ]);
    expect(rows.map(({ cells }) => cells.map(({ delta }) => delta))).toEqual([
      [undefined, undefined],
      [undefined, undefined],
      [undefined, 1.5],
      [undefined, 100],
    ]);
    expect(rows.find(({ assessmentInfo }) => assessmentInfo.name === 'baselineOnly')?.cells[1].scorer).toBeUndefined();
    expect(rows.find(({ assessmentInfo }) => assessmentInfo.name === 'candidateOnly')?.cells[0].scorer).toBeUndefined();
  });

  test('does not merge distinct scorer identities with the same display label', () => {
    const run = createRun('run', { first: 1, second: 2 });
    run.scorers.forEach((scorer) => {
      scorer.assessmentInfo.displayName = 'Same label';
    });
    const rows = buildEvalRunComparisonScorerRows([run]);
    expect(rows.map(({ assessmentInfo }) => assessmentInfo.name)).toEqual(['first', 'second']);
    expect(new Set(rows.map(({ id }) => id)).size).toBe(2);
  });

  test('does not compare categorical scores or incompatible types', () => {
    const rows = buildEvalRunComparisonScorerRows([
      createRun('baseline', { quality: true, category: 'good' }),
      createRun('candidate', { quality: 1, category: 'great' }),
    ]);
    expect(rows.map(({ cells }) => cells.map(({ delta, isBest }) => ({ delta, isBest })))).toEqual([
      [
        { delta: undefined, isBest: false },
        { delta: undefined, isBest: false },
      ],
      [
        { delta: undefined, isBest: false },
        { delta: undefined, isBest: false },
      ],
    ]);
  });

  test('highlights every tied best pass rate, but never ranks numeric values', () => {
    const rows = buildEvalRunComparisonScorerRows([
      createRun('baseline', { quality: true, numeric: 1 }),
      createRun('candidate', { quality: 'yes', numeric: 2 }),
      createRun('third', { quality: false, numeric: 3 }),
    ]);
    expect(
      rows.find(({ assessmentInfo }) => assessmentInfo.name === 'quality')?.cells.map(({ isBest }) => isBest),
    ).toEqual([true, true, false]);
    expect(
      rows.find(({ assessmentInfo }) => assessmentInfo.name === 'numeric')?.cells.map(({ isBest }) => isBest),
    ).toEqual([false, false, false]);
  });

  test.each<EvalRunComparisonScorerRun['status']>(['loading', 'error', 'unavailable'])(
    'withholds deltas and ranking for a %s run',
    (status) => {
      const candidate = { ...createRun('candidate', { quality: true }), status };
      const [row] = buildEvalRunComparisonScorerRows([createRun('baseline', { quality: false }), candidate]);
      expect(row.cells.map(({ delta, isBest }) => ({ delta, isBest }))).toEqual([
        { delta: undefined, isBest: false },
        { delta: undefined, isBest: false },
      ]);
    },
  );

  test('withholds global ranking when a scorer is absent in one run', () => {
    const [row] = buildEvalRunComparisonScorerRows([
      createRun('baseline', { quality: false }),
      createRun('candidate', { quality: true }),
      createRun('empty', {}),
    ]);
    expect(row.cells.map(({ isBest }) => isBest)).toEqual([false, false, false]);
    expect(row.cells.map(({ delta }) => delta)).toEqual([undefined, 100, undefined]);
  });

  test.each([0, 1, 2])('hides comparisons involving truncated run %s and all highest-rate highlights', (index) => {
    const runs = [
      createRun('a', { quality: true }),
      createRun('b', { quality: false }),
      createRun('c', { quality: true }),
    ];
    runs[index].isTruncated = true;
    const [row] = buildEvalRunComparisonScorerRows(runs);
    expect(row.cells.map(({ isBest }) => isBest)).toEqual([false, false, false]);
    expect(row.cells.map(({ delta }) => delta)).toEqual([
      undefined,
      index === 0 || index === 1 ? undefined : -100,
      index === 0 || index === 2 ? undefined : 0,
    ]);
  });

  test('reverses deltas but preserves row order when the baseline changes', () => {
    const runs = [createRun('a', { quality: false, numeric: 2 }), createRun('b', { quality: true, numeric: 3 })];
    const rows = buildEvalRunComparisonScorerRows(runs);
    const swapped = buildEvalRunComparisonScorerRows([...runs].reverse());
    expect(swapped.map(({ id }) => id)).toEqual(rows.map(({ id }) => id));
    expect(swapped.map(({ cells }) => cells[1].delta)).toEqual([-1, -100]);
  });
});
