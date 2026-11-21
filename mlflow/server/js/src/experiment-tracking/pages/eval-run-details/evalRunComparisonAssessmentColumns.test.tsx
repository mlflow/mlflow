import { describe, expect, test } from '@jest/globals';
import { render, screen } from '@databricks/web-shared/test-utils/render';
import { createIntl } from 'react-intl';
import type { FeedbackAssessment, ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import type { TraceComparisonRow, TraceTableColumn } from '@databricks/web-shared/traces-table';
import { EvalRunComparisonTestWrapper } from './EvalRunComparisonTestProviders';
import {
  makeFeedbackAssessment,
  makeTrace,
} from '../../components/experiment-page/components/traces-v4/test-utils/mockTraces';
import { buildAssessmentColumnDefs } from '../../components/experiment-page/components/traces-v4/utils/buildAssessmentColumnDefs';
import {
  getEvalRunComparisonAssessmentDelta,
  withEvalRunComparisonAssessmentDeltas,
} from './evalRunComparisonAssessmentColumns';

const intl = createIntl({ locale: 'en', messages: {} });

const scoredTrace = (id: string, value: number | string, over: Partial<FeedbackAssessment> = {}): ModelTraceInfoV3 =>
  makeTrace(id, { assessments: [makeFeedbackAssessment('quality', value, over)] });

const comparisonRow = (id: string, traces: ModelTraceInfoV3[]): TraceComparisonRow => ({
  id,
  input: id,
  runs: traces.map((trace, index) => ({ runUuid: `run-${index}`, label: `Run ${index}`, color: 'blue', trace })),
});

const qualityColumn = (traces: ModelTraceInfoV3[]): TraceTableColumn => {
  const [column] = buildAssessmentColumnDefs([{ name: 'quality', type: 'numeric' }]);
  return column;
};

const getComparisonRenderer = (column: TraceTableColumn) => {
  if (!column.renderComparisonCell) {
    throw new Error(`Expected ${column.id} to render comparison cells`);
  }
  return column.renderComparisonCell;
};

describe('getEvalRunComparisonAssessmentDelta', () => {
  test.each([
    { direction: 'increase', baselineValue: 0.65, value: 0.7, difference: 0.05 },
    { direction: 'decrease', baselineValue: 0.8, value: 0.6, difference: -0.2 },
  ])('returns the absolute $direction from the baseline score', ({ baselineValue, value, difference }) => {
    expect(
      getEvalRunComparisonAssessmentDelta(
        scoredTrace('candidate', value),
        scoredTrace('baseline', baselineValue),
        'quality',
      ),
    ).toEqual({ value, baselineValue, difference: expect.closeTo(difference, 10) });
  });

  test('returns nothing when the candidate trace lacks the assessment', () => {
    expect(
      getEvalRunComparisonAssessmentDelta(makeTrace('candidate'), scoredTrace('baseline', 0.65), 'quality'),
    ).toBeUndefined();
  });

  test('returns nothing when the baseline run has no trace for the input', () => {
    expect(getEvalRunComparisonAssessmentDelta(scoredTrace('candidate', 0.7), undefined, 'quality')).toBeUndefined();
  });

  test('returns nothing when an assessment errored, even if it kept a value', () => {
    const errored = scoredTrace('candidate', 0.7, {
      feedback: { value: 0.7, error: { error_code: 'JUDGE_ERROR', error_message: 'judge failed' } },
    });
    expect(getEvalRunComparisonAssessmentDelta(errored, scoredTrace('baseline', 0.65), 'quality')).toBeUndefined();
  });

  test('returns nothing for a non-numeric assessment value', () => {
    expect(
      getEvalRunComparisonAssessmentDelta(scoredTrace('candidate', 'yes'), scoredTrace('baseline', 0.65), 'quality'),
    ).toBeUndefined();
  });

  test('returns the difference from a zero baseline score', () => {
    expect(
      getEvalRunComparisonAssessmentDelta(scoredTrace('candidate', 0.5), scoredTrace('baseline', 0), 'quality'),
    ).toEqual({ value: 0.5, baselineValue: 0, difference: 0.5 });
  });

  test.each([
    { change: 'no change', value: 0.8 },
    { change: 'a difference that rounds to 0.00', value: 0.804 },
  ])('returns nothing for $change', ({ value }) => {
    expect(
      getEvalRunComparisonAssessmentDelta(scoredTrace('candidate', value), scoredTrace('baseline', 0.8), 'quality'),
    ).toBeUndefined();
  });
});

describe('withEvalRunComparisonAssessmentDeltas', () => {
  test('adds a comparison renderer to a numeric scorer column and widens it to fit the page delta', () => {
    const traces = [scoredTrace('baseline', 0.65), scoredTrace('candidate', 0.7)];
    const original = qualityColumn(traces);
    const originalSize = original.size;
    const originalMaxSize = original.maxSize;

    const [column] = withEvalRunComparisonAssessmentDeltas([original], traces, [comparisonRow('row', traces)], intl);

    expect(column.renderComparisonCell).toEqual(expect.any(Function));
    expect(column.size).toBeGreaterThan(originalSize ?? 0);
    expect(column.maxSize).toBe(Math.max(originalMaxSize ?? 0, column.size ?? 0));
    expect(original.renderComparisonCell).toBeUndefined();
    expect(original.size).toBe(originalSize);
    expect(original.maxSize).toBe(originalMaxSize);
  });

  test.each([
    { values: 'string', baselineValue: 'yes', candidateValue: 'no' },
    { values: 'integer-only', baselineValue: 1, candidateValue: 3 },
  ])('returns a $values scorer column unchanged', ({ baselineValue, candidateValue }) => {
    const traces = [scoredTrace('baseline', baselineValue), scoredTrace('candidate', candidateValue)];
    const original = qualityColumn(traces);

    const [column] = withEvalRunComparisonAssessmentDeltas([original], traces, [comparisonRow('row', traces)], intl);

    expect(column).toBe(original);
  });

  test('returns non-assessment columns unchanged', () => {
    const traces = [scoredTrace('baseline', 0.65), scoredTrace('candidate', 0.7)];
    const durationColumn: TraceTableColumn = { id: 'duration', header: 'Duration', cell: () => '1s' };
    const accessorColumn: TraceTableColumn = { accessorKey: 'trace_id', header: 'Trace' };

    const columns = withEvalRunComparisonAssessmentDeltas(
      [durationColumn, accessorColumn],
      traces,
      [comparisonRow('row', traces)],
      intl,
    );

    expect(columns[0]).toBe(durationColumn);
    expect(columns[1]).toBe(accessorColumn);
  });

  test('keeps a numeric scorer column at its width when the page has no delta to show', () => {
    const traces = [scoredTrace('baseline', 0.65), scoredTrace('candidate', 0.65)];
    const original = qualityColumn(traces);

    const [column] = withEvalRunComparisonAssessmentDeltas([original], traces, [comparisonRow('row', traces)], intl);

    expect(column.renderComparisonCell).toEqual(expect.any(Function));
    expect(column.size).toBe(original.size);
    expect(column.maxSize).toBe(original.maxSize);
  });

  test('compares every run slot with the baseline slot rather than its neighbor', () => {
    const baseline = scoredTrace('baseline', 0.5);
    const traces = [baseline, scoredTrace('run-a', 0.6), scoredTrace('run-b', 0.75)];
    const row = comparisonRow('row', traces);
    const renderComparisonCell = getComparisonRenderer(
      withEvalRunComparisonAssessmentDeltas([qualityColumn(traces)], traces, [row], intl)[0],
    );

    render(
      <>
        {renderComparisonCell({ trace: traces[2], row, runIndex: 2 })}
        {renderComparisonCell({ trace: baseline, row, runIndex: 0 })}
      </>,
      { wrapper: EvalRunComparisonTestWrapper },
    );

    expect(screen.getByText('+0.25')).toBeVisible();
    expect(screen.queryByText('+0.15')).not.toBeInTheDocument();
    expect(screen.getAllByRole('status')).toHaveLength(1);
    expect(screen.getByText('0.50')).toBeVisible();
  });
});
