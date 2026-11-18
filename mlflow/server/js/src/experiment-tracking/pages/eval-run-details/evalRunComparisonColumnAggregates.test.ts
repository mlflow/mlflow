import { describe, expect, test } from '@jest/globals';
import { createIntl } from 'react-intl';
import {
  KnownEvaluationResultAssessmentStringValue,
  type AssessmentInfo,
} from '@databricks/web-shared/genai-traces-table';
import { ASSESSMENT_SESSION_METADATA_KEY, type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import type { TraceComparisonRow } from '@databricks/web-shared/traces-table';
import { makeFeedbackAssessment } from '../../components/experiment-page/components/traces-v4/test-utils/mockTraces';
import { createComparisonTrace } from './evalRunComparisonTestUtils';
import {
  getEvalRunComparisonColumnAggregates,
  getEvalRunComparisonOutcome,
  getNumericAggregateValue,
  getPercentDelta,
} from './evalRunComparisonColumnAggregates';

const intl = createIntl({ locale: 'en', messages: {} });

interface AssessmentDirectionCase {
  name: string;
  dtype: AssessmentInfo['dtype'];
  low: number | boolean | string;
  high: number | boolean | string;
  label: 'AVG' | 'TRUE' | 'PASS';
  baseline: string;
  candidate: string;
  delta: string;
}

const makeInfo = (dtype: AssessmentInfo['dtype'], name = 'quality'): AssessmentInfo => ({
  name,
  displayName: name,
  dtype,
  isKnown: false,
  isOverall: false,
  metricName: name,
  isCustomMetric: true,
  isEditable: false,
  isRetrievalAssessment: false,
  uniqueValues: new Set(),
  docsLink: '',
  missingTooltip: '',
  description: '',
});

const makeRows = (tracesByRow: (ModelTraceInfoV3 | undefined)[][]): TraceComparisonRow[] =>
  tracesByRow.map((traces, index) => ({
    id: `row-${index}`,
    input: `input-${index}`,
    runs: traces.map((trace, runIndex) => ({
      runUuid: `run-${runIndex}`,
      label: `Run ${runIndex}`,
      color: 'blue',
      trace,
    })),
  }));

const scoredTrace = (value: number | boolean | string): ModelTraceInfoV3 =>
  createComparisonTrace('run', { quality: value });

const statusTrace = (state: ModelTraceInfoV3['state']): ModelTraceInfoV3 => ({
  ...createComparisonTrace('run', {}),
  state,
});

const metricTrace = (duration: string, tokens: number, cost: number): ModelTraceInfoV3 => ({
  ...createComparisonTrace('run', {}),
  execution_duration: duration,
  trace_metadata: {
    'mlflow.trace.tokenUsage': JSON.stringify({ total_tokens: tokens }),
    'mlflow.trace.cost': JSON.stringify({ total_cost: cost }),
  },
});

describe('getEvalRunComparisonColumnAggregates', () => {
  test('computes percentage differences from raw values for comparison menus', () => {
    const rows = makeRows([[scoredTrace(2), scoredTrace(6)]]);
    const aggregate = getEvalRunComparisonColumnAggregates(rows, [makeInfo('numeric')], intl, 2, undefined, true)[
      'assessment:quality'
    ];
    expect(aggregate.values[1].percentDelta).toBe(200);
    expect(aggregate.values[1].delta).toBe(4);
    expect(getPercentDelta(-1, -2)).toBe(50);
    expect(getPercentDelta(0, 0)).toBe(0);
    expect(getPercentDelta(0.1, 0.1000000000001)).toBe(0);
    expect(Object.is(getPercentDelta(0.999999, 1), -0)).toBe(false);
    expect(getPercentDelta(1, 0)).toBeUndefined();
  });

  test('sets percent deltas to zero when absolute deltas round to zero', () => {
    const assessment = getEvalRunComparisonColumnAggregates(
      makeRows([[scoredTrace(1), scoredTrace(1.004)]]),
      [makeInfo('numeric')],
      intl,
      2,
      undefined,
      true,
    )['assessment:quality'].values[1];
    const metric = getEvalRunComparisonColumnAggregates(
      makeRows([[metricTrace('0s', 1, 0), metricTrace('0s', 1.004, 0)]]),
      [],
      intl,
      2,
      undefined,
      true,
    )['tokens'].values[1];

    expect(assessment).toMatchObject({ delta: 0, percentDelta: 0 });
    expect(metric).toMatchObject({ delta: 0, percentDelta: 0 });
  });

  test('averages numeric assessments in baseline-first order with signed baseline differences', () => {
    const rows = makeRows([
      [scoredTrace(0), scoredTrace(8), scoredTrace(1)],
      [scoredTrace(4), scoredTrace(4), undefined],
    ]);

    expect(getEvalRunComparisonColumnAggregates(rows, [makeInfo('numeric')], intl, 3)['assessment:quality']).toEqual({
      label: 'AVG',
      values: [
        { displayValue: '2', displayDelta: undefined },
        { displayValue: '6', displayDelta: '+4', changeCounts: { improved: 1, regressed: 0 } },
        { displayValue: '1', displayDelta: '-1', changeCounts: { improved: 1, regressed: 0 } },
      ],
    });

    const reversedBaseline = makeRows([
      [scoredTrace(8), scoredTrace(0), scoredTrace(1)],
      [scoredTrace(4), scoredTrace(4), undefined],
    ]);
    expect(
      getEvalRunComparisonColumnAggregates(reversedBaseline, [makeInfo('numeric')], intl, 3)['assessment:quality']
        .values,
    ).toEqual([
      { displayValue: '6', displayDelta: undefined },
      { displayValue: '2', displayDelta: '-4', changeCounts: { improved: 0, regressed: 1 } },
      { displayValue: '1', displayDelta: '-5', changeCounts: { improved: 0, regressed: 1 } },
    ]);
  });

  test('computes numeric differences before rounding the displayed averages', () => {
    const rows = makeRows([[scoredTrace(1.004), scoredTrace(1.016)]]);

    expect(
      getEvalRunComparisonColumnAggregates(rows, [makeInfo('numeric')], intl, 2)['assessment:quality'].values,
    ).toEqual([
      { displayValue: '1', displayDelta: undefined },
      { displayValue: '1.02', displayDelta: '+0.01', changeCounts: { improved: 1, regressed: 0 } },
    ]);
  });

  test.each([
    { method: 'mean' as const, baseline: '4.67', candidate: '4.67', delta: undefined },
    { method: 'median' as const, baseline: '3', candidate: '5', delta: '+2' },
    { method: 'min' as const, baseline: '1', candidate: '0', delta: '-1' },
    { method: 'max' as const, baseline: '10', candidate: '9', delta: '-1' },
  ])('supports the $method numeric aggregate for assessment columns', ({ method, baseline, candidate, delta }) => {
    const rows = makeRows([
      [scoredTrace(1), scoredTrace(9)],
      [scoredTrace(3), scoredTrace(5)],
      [scoredTrace(10), scoredTrace(0)],
    ]);

    expect(
      getEvalRunComparisonColumnAggregates(rows, [makeInfo('numeric')], intl, 2, { 'assessment:quality': method })[
        'assessment:quality'
      ],
    ).toEqual({
      label: method === 'mean' ? 'AVG' : method.toUpperCase(),
      values: [
        { displayValue: baseline, displayDelta: undefined },
        { displayValue: candidate, displayDelta: delta, changeCounts: { improved: 2, regressed: 1 } },
      ],
    });
  });

  test('keeps built-in metrics at the default mean even if a metric aggregate method is supplied', () => {
    const rows = makeRows([
      [metricTrace('1s', 4, 0.1), metricTrace('9s', 9, 0.9)],
      [metricTrace('3s', 4, 0.3), metricTrace('5s', 5, 0.5)],
      [metricTrace('10s', 4, 1), metricTrace('0s', 0, 0)],
    ]);

    expect(getEvalRunComparisonColumnAggregates(rows, [], intl, 2, { duration: 'median' })['duration']).toEqual({
      label: 'AVG',
      values: [
        { displayValue: '4.7s', displayDelta: undefined },
        { displayValue: '4.7s', displayDelta: undefined, changeCounts: { improved: 1, regressed: 2 } },
      ],
    });
  });

  test.each(['en', 'ar', 'fa', 'fr'])(
    'suppresses rounded-zero numeric differences in %s without hiding counts or nonzero differences',
    (locale) => {
      const rows = makeRows([[0, 0.004, -0.004, 0.005, -0.005].map(scoredTrace)]);
      const values = getEvalRunComparisonColumnAggregates(
        rows,
        [makeInfo('numeric')],
        createIntl({ locale, messages: {} }),
        5,
      )['assessment:quality'].values;

      expect(values.map((value) => value.displayDelta)).toEqual([undefined, undefined, undefined, '+0.01', '-0.01']);
      expect(values.map((value) => value.changeCounts)).toEqual([
        undefined,
        { improved: 1, regressed: 0 },
        { improved: 0, regressed: 1 },
        { improved: 1, regressed: 0 },
        { improved: 0, regressed: 1 },
      ]);
    },
  );

  test.each(
    (['boolean', 'pass-fail'] as const).flatMap((dtype) =>
      ['en', 'ar', 'fa', 'fr'].map((locale) => ({ dtype, locale })),
    ),
  )(
    'suppresses rounded-zero $dtype rate differences in $locale without hiding counts or nonzero differences',
    ({ dtype, locale }) => {
      const positive = dtype === 'boolean' ? true : KnownEvaluationResultAssessmentStringValue.YES;
      const negative = dtype === 'boolean' ? false : KnownEvaluationResultAssessmentStringValue.NO;
      const rows = makeRows(
        Array.from({ length: 1000 }, (_, index) =>
          [500, 499, 501, 490, 510].map((passingCount) => scoredTrace(index < passingCount ? positive : negative)),
        ),
      );
      const values = getEvalRunComparisonColumnAggregates(
        rows,
        [makeInfo(dtype)],
        createIntl({ locale, messages: {} }),
        5,
      )['assessment:quality'].values;

      expect(values).toEqual([
        { displayValue: '50%', displayDelta: undefined },
        { displayValue: '50%', displayDelta: undefined, changeCounts: { improved: 0, regressed: 1 } },
        { displayValue: '50%', displayDelta: undefined, changeCounts: { improved: 1, regressed: 0 } },
        { displayValue: '49%', displayDelta: '-1', changeCounts: { improved: 0, regressed: 10 } },
        { displayValue: '51%', displayDelta: '+1', changeCounts: { improved: 10, regressed: 0 } },
      ]);
    },
  );

  test.each<AssessmentDirectionCase>(
    ['error', 'error_rate'].flatMap<AssessmentDirectionCase>((name) => [
      { name, dtype: 'numeric', low: 0, high: 6, label: 'AVG', baseline: '4', candidate: '2', delta: '-2' },
      {
        name,
        dtype: 'boolean',
        low: false,
        high: true,
        label: 'TRUE',
        baseline: '67%',
        candidate: '33%',
        delta: '-33',
      },
      {
        name,
        dtype: 'pass-fail',
        low: KnownEvaluationResultAssessmentStringValue.NO,
        high: KnownEvaluationResultAssessmentStringValue.YES,
        label: 'PASS',
        baseline: '67%',
        candidate: '33%',
        delta: '-33',
      },
    ]),
  )(
    'treats higher $dtype $name assessments as improvements regardless of the name',
    ({ name, dtype, low, high, label, baseline, candidate, delta }) => {
      const lowTrace = createComparisonTrace('run', { [name]: low });
      const highTrace = createComparisonTrace('run', { [name]: high });
      const rows = makeRows([
        [highTrace, lowTrace],
        [lowTrace, highTrace],
        [highTrace, lowTrace],
      ]);

      expect(
        getEvalRunComparisonColumnAggregates(rows, [makeInfo(dtype, name)], intl, 2)[`assessment:${name}`],
      ).toEqual({
        label,
        values: [
          { displayValue: baseline, displayDelta: undefined },
          { displayValue: candidate, displayDelta: delta, changeCounts: { improved: 1, regressed: 2 } },
        ],
      });
    },
  );

  test.each(['quality', 'Error', 'ERROR_RATE', 'error_free_score'])(
    'treats higher values as improvements for assessment %s',
    (name) => {
      const rows = makeRows([
        [createComparisonTrace('baseline', { [name]: 1 }), createComparisonTrace('candidate', { [name]: 2 })],
      ]);

      expect(
        getEvalRunComparisonColumnAggregates(rows, [makeInfo('numeric', name)], intl, 2)[`assessment:${name}`],
      ).toEqual({
        label: 'AVG',
        values: [
          { displayValue: '1', displayDelta: undefined },
          { displayValue: '2', displayDelta: '+1', changeCounts: { improved: 1, regressed: 0 } },
        ],
      });
    },
  );

  test('summarizes completed trace failures and counts changes only for completed matched pairs', () => {
    const rows = makeRows([
      [statusTrace('ERROR'), statusTrace('OK'), statusTrace('ERROR')],
      [statusTrace('OK'), statusTrace('ERROR'), statusTrace('ERROR')],
      [statusTrace('ERROR'), statusTrace('OK'), undefined],
      [statusTrace('OK'), statusTrace('OK'), statusTrace('ERROR')],
      [statusTrace('IN_PROGRESS'), statusTrace('ERROR'), statusTrace('STATE_UNSPECIFIED')],
      [statusTrace('STATE_UNSPECIFIED'), statusTrace('IN_PROGRESS'), statusTrace('ERROR')],
      [undefined, statusTrace('ERROR'), statusTrace('IN_PROGRESS')],
      [statusTrace('ERROR'), undefined, undefined],
    ]);

    expect(getEvalRunComparisonColumnAggregates(rows, [], intl, 3)['state']).toEqual({
      label: 'ERROR RATE',
      values: [
        { displayValue: '60%', displayDelta: undefined },
        { displayValue: '50%', displayDelta: '-10', changeCounts: { improved: 2, regressed: 1 } },
        { displayValue: '100%', displayDelta: '+40', changeCounts: { improved: 0, regressed: 2 } },
      ],
    });
  });

  test('omits status deltas and counts when the baseline has no completed traces', () => {
    const rows = makeRows([
      [statusTrace('IN_PROGRESS'), statusTrace('ERROR')],
      [statusTrace('STATE_UNSPECIFIED'), statusTrace('OK')],
    ]);

    expect(getEvalRunComparisonColumnAggregates(rows, [], intl, 2)['state']).toEqual({
      label: 'ERROR RATE',
      values: [
        { displayValue: '—', displayDelta: undefined },
        { displayValue: '50%', displayDelta: undefined },
      ],
    });
  });

  test('suppresses rounded-zero error-rate differences without suppressing matched improvements', () => {
    const rows = makeRows([
      [statusTrace('ERROR'), statusTrace('OK')],
      ...Array.from({ length: 200 }, () => [statusTrace('OK'), statusTrace('OK')]),
    ]);

    expect(getEvalRunComparisonColumnAggregates(rows, [], intl, 2)['state']).toEqual({
      label: 'ERROR RATE',
      values: [
        { displayValue: '0%', displayDelta: undefined },
        { displayValue: '0%', displayDelta: undefined, changeCounts: { improved: 1, regressed: 0 } },
      ],
    });
  });

  test('excludes scorer failures from scores without marking successful traces as failed', () => {
    const failedScore = makeFeedbackAssessment('error', 100, {
      error: { error_code: 'FAILED', error_message: 'Scorer failed' },
    });
    const rows = makeRows([
      [
        { ...statusTrace('OK'), assessments: [failedScore] },
        { ...statusTrace('OK'), assessments: [makeFeedbackAssessment('error', 0.1)] },
      ],
      [
        { ...statusTrace('ERROR'), assessments: [makeFeedbackAssessment('error', 0.6)] },
        { ...statusTrace('OK'), assessments: [failedScore] },
      ],
    ]);
    const aggregates = getEvalRunComparisonColumnAggregates(rows, [makeInfo('numeric', 'error')], intl, 2);

    expect(aggregates['assessment:error']).toEqual({
      label: 'AVG',
      values: [
        { displayValue: '0.6', displayDelta: undefined },
        { displayValue: '0.1', displayDelta: '-0.5' },
      ],
    });
    expect(aggregates['state']).toEqual({
      label: 'ERROR RATE',
      values: [
        { displayValue: '50%', displayDelta: undefined },
        { displayValue: '0%', displayDelta: '-50', changeCounts: { improved: 1, regressed: 0 } },
      ],
    });
  });

  test('excludes same-name session feedback from trace averages and improved or regressed counts', () => {
    const scores = [
      { baseline: 0.65, candidate: 0.7, session: 0.88 },
      { baseline: 0.72, candidate: 0.78 },
      { baseline: 0.79, candidate: 0.86 },
      { baseline: 0.86, candidate: 0.94 },
      { baseline: 0.93, candidate: 0.7 },
      { baseline: 0.65, candidate: 0.78 },
      { baseline: 0.72, candidate: 0.86, session: 0.94 },
      { baseline: 0.79, candidate: 0.94 },
      { baseline: 0.86, candidate: 0.7 },
      { baseline: 0.93, candidate: 0.78 },
      { baseline: 0.65, candidate: 0.86 },
    ];
    const rows = makeRows(
      scores.map(({ baseline, candidate, session }) => [
        {
          ...scoredTrace(baseline),
          assessments: [
            makeFeedbackAssessment('quality', baseline),
            ...(session === undefined
              ? []
              : [
                  makeFeedbackAssessment('quality', session, {
                    metadata: { [ASSESSMENT_SESSION_METADATA_KEY]: 'session-1' },
                  }),
                ]),
          ],
        },
        scoredTrace(candidate),
      ]),
    );

    expect(getEvalRunComparisonColumnAggregates(rows, [makeInfo('numeric')], intl, 2)['assessment:quality']).toEqual({
      label: 'AVG',
      values: [
        { displayValue: '0.78', displayDelta: undefined },
        { displayValue: '0.81', displayDelta: '+0.03', changeCounts: { improved: 8, regressed: 3 } },
      ],
    });
  });

  test('treats session-only feedback as missing trace samples without a baseline delta or change count', () => {
    const sessionOnlyTrace: ModelTraceInfoV3 = {
      ...scoredTrace(0.88),
      assessments: [
        makeFeedbackAssessment('quality', 0.88, { metadata: { [ASSESSMENT_SESSION_METADATA_KEY]: 'session-1' } }),
      ],
    };
    const rows = makeRows([[sessionOnlyTrace, scoredTrace(0.7), sessionOnlyTrace]]);

    expect(getEvalRunComparisonColumnAggregates(rows, [makeInfo('numeric')], intl, 3)['assessment:quality']).toEqual({
      label: 'AVG',
      values: [{ displayValue: '—' }, { displayValue: '0.7', displayDelta: undefined }, { displayValue: '—' }],
    });
  });

  test('averages duration, token and cost samples while retaining zero and excluding invalid metrics', () => {
    const invalidMetrics: ModelTraceInfoV3 = {
      ...createComparisonTrace('run', {}),
      execution_duration: 'invalid',
      trace_metadata: {
        'mlflow.trace.tokenUsage': '{"total_tokens":1e999}',
        'mlflow.trace.cost': '{"total_cost":1e999}',
      },
    };
    const rows = makeRows([
      [metricTrace('0s', 0, 0), metricTrace('500ms', 3, 0.006), metricTrace('3s', 9, 0.03)],
      [metricTrace('2s', 10, 0.02), undefined, undefined],
      [invalidMetrics, undefined, undefined],
      [metricTrace('-2s', -50, -1), undefined, undefined],
    ]);
    const aggregates = getEvalRunComparisonColumnAggregates(rows, [], intl, 3);

    expect(aggregates['duration']).toEqual({
      label: 'AVG',
      values: [
        { displayValue: '1.0s', displayDelta: undefined },
        { displayValue: '500ms', displayDelta: '-500', changeCounts: { improved: 0, regressed: 1 } },
        { displayValue: '3.0s', displayDelta: '+2.0', changeCounts: { improved: 0, regressed: 1 } },
      ],
    });
    expect(aggregates['tokens']).toEqual({
      label: 'AVG',
      values: [
        { displayValue: '5', displayDelta: undefined },
        { displayValue: '3', displayDelta: '-2', changeCounts: { improved: 0, regressed: 1 } },
        { displayValue: '9', displayDelta: '+4', changeCounts: { improved: 0, regressed: 1 } },
      ],
    });
    expect(aggregates['cost']).toEqual({
      label: 'AVG',
      values: [
        { displayValue: '$0.01', displayDelta: undefined },
        { displayValue: '$0.006', displayDelta: '-0.004', changeCounts: { improved: 0, regressed: 1 } },
        { displayValue: '$0.03', displayDelta: '+0.02', changeCounts: { improved: 0, regressed: 1 } },
      ],
    });
  });

  test.each([
    { duration: '500ms', displayValue: '500ms', deltaMagnitude: '500' },
    { duration: '2s', displayValue: '2.0s', deltaMagnitude: '2.0' },
    { duration: '120s', displayValue: '2.0min', deltaMagnitude: '2.0' },
    { duration: '7200s', displayValue: '2.0h', deltaMagnitude: '2.0' },
    { duration: '172800s', displayValue: '2.0d', deltaMagnitude: '2.0' },
  ])(
    'omits duration delta units for $duration while preserving value units and precision',
    ({ duration, displayValue, deltaMagnitude }) => {
      const zero = metricTrace('0s', 0, 0);
      const trace = metricTrace(duration, 0, 0);
      expect(getEvalRunComparisonColumnAggregates(makeRows([[zero, trace]]), [], intl, 2)['duration'].values).toEqual([
        { displayValue: '0ms', displayDelta: undefined },
        { displayValue, displayDelta: `+${deltaMagnitude}`, changeCounts: { improved: 0, regressed: 1 } },
      ]);
      expect(getEvalRunComparisonColumnAggregates(makeRows([[trace, zero]]), [], intl, 2)['duration'].values).toEqual([
        { displayValue, displayDelta: undefined },
        { displayValue: '0ms', displayDelta: `-${deltaMagnitude}`, changeCounts: { improved: 1, regressed: 0 } },
      ]);
    },
  );

  test.each([
    { cost: 0.000001, displayValue: '$0.000001', displayDelta: '+0.000001' },
    { cost: 1234.56, displayValue: '$1,234.56', displayDelta: '+1,234.56' },
  ])(
    'omits the currency symbol from a $cost delta without losing precision or separators',
    ({ cost, displayValue, displayDelta }) => {
      const rows = makeRows([[metricTrace('0s', 0, 0), metricTrace('0s', 0, cost)]]);
      expect(getEvalRunComparisonColumnAggregates(rows, [], intl, 2)['cost'].values).toEqual([
        { displayValue: '$0.00', displayDelta: undefined },
        { displayValue, displayDelta, changeCounts: { improved: 0, regressed: 1 } },
      ]);
    },
  );

  test.each<{
    dtype: AssessmentInfo['dtype'];
    label: 'PASS' | 'TRUE';
    positive: string | boolean;
    negative: string | boolean;
  }>([
    { dtype: 'boolean', label: 'TRUE', positive: true, negative: false },
    {
      dtype: 'pass-fail',
      label: 'PASS',
      positive: KnownEvaluationResultAssessmentStringValue.YES,
      negative: KnownEvaluationResultAssessmentStringValue.NO,
    },
  ])(
    'uses only valid $dtype samples for rates and unitless percentage-point differences',
    ({ dtype, label, positive, negative }) => {
      const errored: ModelTraceInfoV3 = {
        ...scoredTrace(positive),
        assessments: [
          makeFeedbackAssessment('quality', positive, {
            error: { error_code: 'FAILED', error_message: 'Scorer failed' },
          }),
        ],
      };
      const rows = makeRows([
        [scoredTrace(negative), scoredTrace(positive), scoredTrace(negative)],
        [scoredTrace(positive), undefined, undefined],
        [scoredTrace('not a rate'), errored, undefined],
      ]);

      expect(getEvalRunComparisonColumnAggregates(rows, [makeInfo(dtype)], intl, 3)['assessment:quality']).toEqual({
        label,
        values: [
          { displayValue: '50%', displayDelta: undefined },
          { displayValue: '100%', displayDelta: '+50', changeCounts: { improved: 1, regressed: 0 } },
          { displayValue: '0%', displayDelta: '-50' },
        ],
      });
    },
  );

  test('excludes missing, invalidated, errored, incompatible and non-finite assessment samples', () => {
    const invalidated = {
      ...scoredTrace(200),
      assessments: [makeFeedbackAssessment('quality', 200, { valid: false })],
    };
    const errored = {
      ...scoredTrace(100),
      assessments: [
        makeFeedbackAssessment('quality', 100, { error: { error_code: 'FAILED', error_message: 'Scorer failed' } }),
      ],
    };
    const rows = makeRows([
      [scoredTrace(0), invalidated],
      [scoredTrace(4), errored],
      [invalidated, scoredTrace(Number.NaN)],
      [errored, scoredTrace(Number.POSITIVE_INFINITY)],
      [scoredTrace(Number.NaN), undefined],
      [scoredTrace(Number.NEGATIVE_INFINITY), undefined],
      [scoredTrace('7'), undefined],
      [createComparisonTrace('run', {}), undefined],
    ]);

    expect(getEvalRunComparisonColumnAggregates(rows, [makeInfo('numeric')], intl, 2)['assessment:quality']).toEqual({
      label: 'AVG',
      values: [{ displayValue: '2', displayDelta: undefined }, { displayValue: '—' }],
    });
  });

  test('shows missing values for every selected run when there are no retained rows', () => {
    const aggregates = getEvalRunComparisonColumnAggregates([], [makeInfo('numeric')], intl, 3);

    expect(aggregates['assessment:quality']).toEqual({
      label: 'AVG',
      values: [{ displayValue: '—' }, { displayValue: '—' }, { displayValue: '—' }],
    });
    for (const column of ['duration', 'tokens', 'cost']) {
      expect(aggregates[column]).toEqual({
        label: 'AVG',
        values: Array.from({ length: 3 }, () => ({ displayValue: '—', displayDelta: undefined })),
      });
    }
    expect(aggregates['state']).toEqual({
      label: 'ERROR RATE',
      values: Array.from({ length: 3 }, () => ({ displayValue: '—', displayDelta: undefined })),
    });
  });

  test.each<AssessmentInfo['dtype']>(['numeric', 'boolean', 'pass-fail'])(
    'suppresses %s deltas when the baseline has no valid sample',
    (dtype) => {
      const value =
        dtype === 'numeric' ? 7 : dtype === 'boolean' ? true : KnownEvaluationResultAssessmentStringValue.YES;
      const rows = makeRows([[undefined, scoredTrace(value)]]);

      expect(
        getEvalRunComparisonColumnAggregates(rows, [makeInfo(dtype)], intl, 2)['assessment:quality'].values,
      ).toEqual([{ displayValue: '—' }, { displayValue: dtype === 'numeric' ? '7' : '100%', displayDelta: undefined }]);
    },
  );

  test('suppresses differences for unchanged numeric and standard metric averages', () => {
    const trace = { ...metricTrace('2s', 4, 0.01), assessments: [makeFeedbackAssessment('quality', 4)] };
    const aggregates = getEvalRunComparisonColumnAggregates(makeRows([[trace, trace]]), [makeInfo('numeric')], intl, 2);

    expect(Object.values(aggregates).map((aggregate) => aggregate.values.map((value) => value.displayDelta))).toEqual([
      [undefined, undefined],
      [undefined, undefined],
      [undefined, undefined],
      [undefined, undefined],
      [undefined, undefined],
    ]);

    const roundedRows = makeRows([
      [
        { ...metricTrace('1.0001s', 0.1, 0.0000001), assessments: [makeFeedbackAssessment('quality', 1.004)] },
        { ...metricTrace('1.00025s', 0.15, 0.00000015), assessments: [makeFeedbackAssessment('quality', 1.006)] },
      ],
      [
        { ...metricTrace('1.0003s', 0.2, 0.0000002), assessments: [makeFeedbackAssessment('quality', 1.004)] },
        undefined,
      ],
    ]);
    const roundedAggregates = getEvalRunComparisonColumnAggregates(roundedRows, [makeInfo('numeric')], intl, 2);
    expect(
      Object.values(roundedAggregates).map((aggregate) => aggregate.values.map((value) => value.displayDelta)),
    ).toEqual([
      [undefined, undefined],
      [undefined, undefined],
      [undefined, undefined],
      [undefined, undefined],
      [undefined, undefined],
    ]);
  });

  test('shows comparison metrics without differences when the baseline metrics are absent', () => {
    const rows = makeRows([[undefined, metricTrace('2s', 4, 0.01)]]);

    expect(getEvalRunComparisonColumnAggregates(rows, [], intl, 2)).toEqual({
      state: {
        label: 'ERROR RATE',
        values: [
          { displayValue: '—', displayDelta: undefined },
          { displayValue: '0%', displayDelta: undefined },
        ],
      },
      duration: {
        label: 'AVG',
        values: [
          { displayValue: '—', displayDelta: undefined },
          { displayValue: '2.0s', displayDelta: undefined },
        ],
      },
      tokens: {
        label: 'AVG',
        values: [
          { displayValue: '—', displayDelta: undefined },
          { displayValue: '4', displayDelta: undefined },
        ],
      },
      cost: {
        label: 'AVG',
        values: [
          { displayValue: '—', displayDelta: undefined },
          { displayValue: '$0.01', displayDelta: undefined },
        ],
      },
    });
  });

  test('counts distinct valid categorical values per run without a label or difference', () => {
    const errored = {
      ...scoredTrace('invalid'),
      assessments: [
        makeFeedbackAssessment('quality', 'invalid', { error: { error_code: 'FAILED', error_message: 'Failed' } }),
      ],
    };
    const rows = makeRows([
      [scoredTrace('red'), scoredTrace('green'), undefined],
      [scoredTrace('blue'), scoredTrace('green'), undefined],
      [scoredTrace('red'), errored, undefined],
    ]);

    expect(getEvalRunComparisonColumnAggregates(rows, [makeInfo('string')], intl, 3)['assessment:quality']).toEqual({
      label: undefined,
      values: [
        { displayValue: '2 values', displayDelta: undefined },
        { displayValue: '1 value', displayDelta: undefined },
        { displayValue: '—' },
      ],
    });
  });

  test('retains separate run-specific assessments for copies of the same trace identifier', () => {
    const rows = makeRows([[scoredTrace(2), scoredTrace(8)]]);

    expect(
      getEvalRunComparisonColumnAggregates(rows, [makeInfo('numeric')], intl, 2)['assessment:quality'].values,
    ).toEqual([
      { displayValue: '2', displayDelta: undefined },
      { displayValue: '8', displayDelta: '+6', changeCounts: { improved: 1, regressed: 0 } },
    ]);
  });

  test('counts improvements and regressions per matched trace rather than the aggregate difference', () => {
    const rows = makeRows([
      [scoredTrace(2), scoredTrace(6), scoredTrace(1)],
      [scoredTrace(8), scoredTrace(0), scoredTrace(9)],
      [scoredTrace(0), scoredTrace(1), scoredTrace(0)],
      [scoredTrace(4), scoredTrace(4), undefined],
      [undefined, scoredTrace(100), scoredTrace(-100)],
      [scoredTrace(Number.NaN), scoredTrace(4), scoredTrace(6)],
      [scoredTrace(4), scoredTrace(Number.POSITIVE_INFINITY), undefined],
    ]);

    expect(
      getEvalRunComparisonColumnAggregates(rows, [makeInfo('numeric')], intl, 3)['assessment:quality'].values.map(
        (value) => value.changeCounts,
      ),
    ).toEqual([undefined, { improved: 2, regressed: 1 }, { improved: 1, regressed: 1 }]);
  });

  test('treats lower matched duration, token and cost samples as improvements', () => {
    const rows = makeRows([
      [metricTrace('1s', 4, 0.1), metricTrace('0s', 0, 0)],
      [metricTrace('1s', 4, 0.1), metricTrace('2s', 8, 0.2)],
      [metricTrace('1s', 4, 0.1), metricTrace('1s', 4, 0.1)],
      [undefined, metricTrace('10s', 10, 10)],
      [metricTrace('1s', 4, 0.1), undefined],
      [metricTrace('-1s', -4, -0.1), metricTrace('1s', 4, 0.1)],
      [metricTrace('1s', 4, 0.1), metricTrace('invalid', -4, -0.1)],
    ]);
    const aggregates = getEvalRunComparisonColumnAggregates(rows, [], intl, 2);

    for (const column of ['duration', 'tokens', 'cost']) {
      expect(aggregates[column].values.map((value) => value.changeCounts)).toEqual([
        undefined,
        { improved: 1, regressed: 1 },
      ]);
    }
  });

  test.each<{ dtype: AssessmentInfo['dtype']; low: number | boolean | string; high: number | boolean | string }>([
    { dtype: 'numeric', low: 0, high: 4 },
    { dtype: 'boolean', low: false, high: true },
    {
      dtype: 'pass-fail',
      low: KnownEvaluationResultAssessmentStringValue.NO,
      high: KnownEvaluationResultAssessmentStringValue.YES,
    },
  ])('counts each matched trace once using its mean $dtype assessment value', ({ dtype, low, high }) => {
    const mixed: ModelTraceInfoV3 = {
      ...scoredTrace(low),
      assessments: [makeFeedbackAssessment('quality', low), makeFeedbackAssessment('quality', high)],
    };
    const rows = makeRows([
      [scoredTrace(low), mixed],
      [scoredTrace(high), mixed],
      [mixed, mixed],
    ]);

    expect(
      getEvalRunComparisonColumnAggregates(rows, [makeInfo(dtype)], intl, 2)['assessment:quality'].values.map(
        (value) => value.changeCounts,
      ),
    ).toEqual([undefined, { improved: 1, regressed: 1 }]);
  });

  test('classifies assessment and metric outcomes relative to the first run', () => {
    const assessmentInfo = makeInfo('numeric');
    const assessmentRow = makeRows([[scoredTrace(2), scoredTrace(4)]])[0];
    expect(getEvalRunComparisonOutcome(assessmentRow, 'assessment:quality', 'run-1', [assessmentInfo])).toBe(
      'improvement',
    );

    const metricRow = makeRows([[metricTrace('2s', 8, 0.2), metricTrace('1s', 4, 0.1)]])[0];
    expect(getEvalRunComparisonOutcome(metricRow, 'duration', 'run-1', [])).toBe('improvement');
    expect(getEvalRunComparisonOutcome(metricRow, 'tokens', 'run-1', [])).toBe('improvement');
    expect(getEvalRunComparisonOutcome(metricRow, 'cost', 'run-1', [])).toBe('improvement');

    const statusRow = makeRows([[statusTrace('ERROR'), statusTrace('OK')]])[0];
    expect(getEvalRunComparisonOutcome(statusRow, 'state', 'run-1', [])).toBe('improvement');
    expect(getNumericAggregateValue([1, 3, 10], 'median')).toBe(3);
  });
});
