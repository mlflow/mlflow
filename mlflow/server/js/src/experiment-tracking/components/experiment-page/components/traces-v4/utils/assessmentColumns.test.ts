import { describe, expect, test } from '@jest/globals';
import { ASSESSMENT_SESSION_METADATA_KEY } from '@databricks/web-shared/model-trace-explorer';
import type { Assessment } from '@databricks/web-shared/model-trace-explorer';
import {
  computeAssessmentColumns,
  extractTraceIssues,
  getAssessmentColumnType,
  getSortableAssessmentNames,
  pickCellAssessment,
} from './assessmentColumns';
import { makeFeedbackAssessment, makeIssueAssessment, makeTrace } from '../test-utils/mockTraces';

const traceWith = (id: string, assessments: Assessment[]) => makeTrace(id, { assessments });

describe('computeAssessmentColumns', () => {
  test('defaults to a column per assessment name on the page, sorted', () => {
    const traces = [
      traceWith('t1', [makeFeedbackAssessment('relevance', 'yes'), makeFeedbackAssessment('safety', 'no')]),
      traceWith('t2', [makeFeedbackAssessment('correctness', 'yes')]),
    ];
    expect(computeAssessmentColumns(traces, {})).toEqual({
      candidateNames: ['correctness', 'relevance', 'safety'],
      visibleNames: ['correctness', 'relevance', 'safety'],
    });
  });

  test('an opt-out override hides the column but keeps it a candidate', () => {
    const traces = [traceWith('t1', [makeFeedbackAssessment('relevance', 'yes')])];
    expect(computeAssessmentColumns(traces, { relevance: false })).toEqual({
      candidateNames: ['relevance'],
      visibleNames: [],
    });
  });

  test('an opted-in name is a visible column even when absent from the page', () => {
    const traces = [traceWith('t1', [makeFeedbackAssessment('relevance', 'yes')])];
    expect(computeAssessmentColumns(traces, { safety: true })).toEqual({
      candidateNames: ['relevance', 'safety'],
      visibleNames: ['relevance', 'safety'],
    });
  });

  test('an opted-out name absent from the page drops out of the candidate set entirely', () => {
    const traces = [traceWith('t1', [makeFeedbackAssessment('relevance', 'yes')])];
    expect(computeAssessmentColumns(traces, { safety: false })).toEqual({
      candidateNames: ['relevance'],
      visibleNames: ['relevance'],
    });
  });

  test('excludes invalid, session-level, notes, and internal-judge assessments', () => {
    const traces = [
      traceWith('t1', [
        makeFeedbackAssessment('relevance', 'yes'),
        makeFeedbackAssessment('invalidOne', 'yes', { valid: false }),
        makeFeedbackAssessment('sessionOne', 'yes', {
          metadata: { [ASSESSMENT_SESSION_METADATA_KEY]: 'session-1' },
        }),
        makeFeedbackAssessment('mlflow.notes', 'a note'),
        makeFeedbackAssessment('_issue_discovery_judge', 'yes'),
      ]),
    ];
    expect(computeAssessmentColumns(traces, {})).toEqual({
      candidateNames: ['relevance'],
      visibleNames: ['relevance'],
    });
  });

  test('excludes issue-reference assessments (they render in the dedicated Issues column)', () => {
    const traces = [
      traceWith('t1', [makeFeedbackAssessment('relevance', 'yes'), makeIssueAssessment('hallucination')]),
    ];
    expect(computeAssessmentColumns(traces, {})).toEqual({
      candidateNames: ['relevance'],
      visibleNames: ['relevance'],
    });
  });

  test('dedupes a name that appears across many traces', () => {
    const traces = [
      traceWith('t1', [makeFeedbackAssessment('relevance', 'yes')]),
      traceWith('t2', [makeFeedbackAssessment('relevance', 'no')]),
    ];
    expect(computeAssessmentColumns(traces, {}).candidateNames).toEqual(['relevance']);
  });

  test('an empty page with no overrides yields no columns', () => {
    expect(computeAssessmentColumns([], {})).toEqual({ candidateNames: [], visibleNames: [] });
  });
});

describe('pickCellAssessment', () => {
  test('returns the most recent assessment for the name', () => {
    const older = makeFeedbackAssessment('relevance', 'no', { create_time: '2025-01-01T00:00:00.000Z' });
    const newer = makeFeedbackAssessment('relevance', 'yes', { create_time: '2025-06-01T00:00:00.000Z' });
    const trace = traceWith('t1', [older, newer]);
    expect(pickCellAssessment(trace, 'relevance')).toBe(newer);
  });

  test('breaks equal-timestamp ties by assessment id', () => {
    const timestamp = '2025-06-01T00:00:00.000Z';
    const lowerId = makeFeedbackAssessment('relevance', 'no', {
      assessment_id: 'assessment-a',
      create_time: timestamp,
    });
    const higherId = makeFeedbackAssessment('relevance', 'yes', {
      assessment_id: 'assessment-z',
      create_time: timestamp,
    });
    const trace = traceWith('t1', [higherId, lowerId]);

    expect(pickCellAssessment(trace, 'relevance')).toBe(higherId);
  });

  test('compares timestamps by instant when fractional-second precision differs', () => {
    const older = makeFeedbackAssessment('relevance', 'no', { create_time: '2025-06-01T00:00:00Z' });
    const newer = makeFeedbackAssessment('relevance', 'yes', { create_time: '2025-06-01T00:00:00.100Z' });
    const trace = traceWith('t1', [newer, older]);

    expect(pickCellAssessment(trace, 'relevance')).toBe(newer);
  });

  test('ignores other names and returns undefined when none match', () => {
    const trace = traceWith('t1', [makeFeedbackAssessment('relevance', 'yes')]);
    expect(pickCellAssessment(trace, 'safety')).toBeUndefined();
  });

  test('skips non-displayable (invalid) assessments', () => {
    const trace = traceWith('t1', [makeFeedbackAssessment('relevance', 'yes', { valid: false })]);
    expect(pickCellAssessment(trace, 'relevance')).toBeUndefined();
  });

  test('does not surface an issue-reference assessment as a cell value', () => {
    const issue = makeIssueAssessment('hallucination');
    const trace = traceWith('t1', [issue]);
    expect(pickCellAssessment(trace, issue.assessment_name)).toBeUndefined();
  });
});

describe('getSortableAssessmentNames', () => {
  test('keeps trace-level feedback and off-page opted-in names', () => {
    const traces = [traceWith('t1', [makeFeedbackAssessment('quality', 0.8)])];

    expect(getSortableAssessmentNames(traces, ['off-page', 'quality'])).toEqual(['off-page', 'quality']);
  });

  test('excludes a blank assessment name', () => {
    const traces = [traceWith('t1', [makeFeedbackAssessment('', 0.8)])];

    expect(getSortableAssessmentNames(traces, ['', 'off-page'])).toEqual(['off-page']);
  });

  test('excludes expectation-only and span-feedback-only columns', () => {
    const expectation: Assessment = {
      assessment_id: 'expectation-1',
      assessment_name: 'expected-answer',
      trace_id: 't1',
      source: { source_type: 'HUMAN', source_id: 'user@example.com' },
      create_time: '2025-01-01T00:00:00.000Z',
      last_update_time: '2025-01-01T00:00:00.000Z',
      expectation: { value: 'answer' },
    };
    const traces = [
      traceWith('t1', [
        expectation,
        makeFeedbackAssessment('span-quality', 0.8, { span_id: 'span-1' }),
        makeFeedbackAssessment('trace-quality', 0.9),
      ]),
    ];

    expect(getSortableAssessmentNames(traces, ['expected-answer', 'span-quality', 'trace-quality'])).toEqual([
      'trace-quality',
    ]);
  });

  test('excludes a mixed column when its displayed assessment is not trace-level feedback', () => {
    const newerExpectation: Assessment = {
      assessment_id: 'expectation-1',
      assessment_name: 'quality',
      trace_id: 't1',
      source: { source_type: 'HUMAN', source_id: 'user@example.com' },
      create_time: '2025-06-01T00:00:00.000Z',
      last_update_time: '2025-06-01T00:00:00.000Z',
      expectation: { value: 'answer' },
    };
    const traces = [
      traceWith('t1', [
        makeFeedbackAssessment('quality', 0.8, { create_time: '2025-01-01T00:00:00.000Z' }),
        newerExpectation,
      ]),
      traceWith('t2', [
        makeFeedbackAssessment('span-quality', 0.7, { create_time: '2025-01-01T00:00:00.000Z' }),
        makeFeedbackAssessment('span-quality', 0.9, {
          create_time: '2025-06-01T00:00:00.000Z',
          span_id: 'span-1',
        }),
      ]),
    ];

    expect(getSortableAssessmentNames(traces, ['quality', 'span-quality'])).toEqual([]);
  });
});

describe('extractTraceIssues', () => {
  test('maps issue-reference assessments to {id, name} using assessment_name as the id', () => {
    const trace = traceWith('t1', [
      makeFeedbackAssessment('relevance', 'yes'),
      makeIssueAssessment('hallucination'),
      makeIssueAssessment('toxicity'),
    ]);
    expect(extractTraceIssues(trace)).toEqual([
      { id: 'issue-hallucination', name: 'hallucination' },
      { id: 'issue-toxicity', name: 'toxicity' },
    ]);
  });

  test('falls back to the assessment name when issue_name is empty', () => {
    const trace = traceWith('t1', [makeIssueAssessment('x', { issue: { issue_name: '' } })]);
    expect(extractTraceIssues(trace)).toEqual([{ id: 'issue-x', name: 'issue-x' }]);
  });

  test('returns no issues for a trace with only feedback assessments', () => {
    const trace = traceWith('t1', [makeFeedbackAssessment('relevance', 'yes')]);
    expect(extractTraceIssues(trace)).toEqual([]);
  });

  test('skips invalid issue-reference assessments, matching the prior tab', () => {
    const trace = traceWith('t1', [
      makeIssueAssessment('hallucination', { valid: false }),
      makeIssueAssessment('toxicity'),
    ]);
    expect(extractTraceIssues(trace)).toEqual([{ id: 'issue-toxicity', name: 'toxicity' }]);
  });
});

describe('getAssessmentColumnType', () => {
  test('returns "numeric" when any value is a non-integer', () => {
    const traces = [
      traceWith('t1', [makeFeedbackAssessment('score', 0.75)]),
      traceWith('t2', [makeFeedbackAssessment('score', 1)]),
    ];
    expect(getAssessmentColumnType(traces, 'score')).toBe('numeric');
  });

  test('returns "categorical" when all values are integers or missing', () => {
    const traces = [
      traceWith('t1', [makeFeedbackAssessment('category', 1)]),
      traceWith('t2', [makeFeedbackAssessment('category', 2)]),
    ];
    expect(getAssessmentColumnType(traces, 'category')).toBe('categorical');
  });

  test('returns "numeric" when mixed integer and non-integer values are present', () => {
    const traces = [
      traceWith('t1', [makeFeedbackAssessment('score', 0.5)]),
      traceWith('t2', [makeFeedbackAssessment('score', 1)]),
    ];
    expect(getAssessmentColumnType(traces, 'score')).toBe('numeric');
  });

  test('returns "categorical" when traces list is empty', () => {
    expect(getAssessmentColumnType([], 'score')).toBe('categorical');
  });

  test('returns "categorical" when no assessments match the name', () => {
    const traces = [traceWith('t1', [makeFeedbackAssessment('other', 'yes')])];
    expect(getAssessmentColumnType(traces, 'score')).toBe('categorical');
  });
});
