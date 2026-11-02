import { describe, expect, test } from '@jest/globals';
import { ASSESSMENT_SESSION_METADATA_KEY } from '@databricks/web-shared/model-trace-explorer';
import type { Assessment } from '@databricks/web-shared/model-trace-explorer';
import {
  computeAssessmentColumns,
  extractTraceIssues,
  getAssessmentColumnSourceType,
  getAssessmentColumnType,
  getSessionAssessmentNames,
  pickCellAssessment,
  pickSessionCellAssessment,
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

  test('a seeded name is a column even when absent from the page, and an opt-out still hides it', () => {
    const traces = [traceWith('t1', [makeFeedbackAssessment('relevance', 'yes')])];
    expect(computeAssessmentColumns(traces, {}, { seededNames: ['safety'] })).toEqual({
      candidateNames: ['relevance', 'safety'],
      visibleNames: ['relevance', 'safety'],
    });
    expect(computeAssessmentColumns(traces, { safety: false }, { seededNames: ['safety'] })).toEqual({
      candidateNames: ['relevance', 'safety'],
      visibleNames: ['relevance'],
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

describe('getAssessmentColumnSourceType', () => {
  const human = { source_type: 'HUMAN' as const, source_id: 'reviewer' };

  test('returns the source shared by every assessment in the column', () => {
    const traces = [
      makeTrace('a', { assessments: [makeFeedbackAssessment('tone', 'yes', { source: human })] }),
      makeTrace('b', { assessments: [makeFeedbackAssessment('tone', 'no', { source: human })] }),
      makeTrace('c'),
    ];

    expect(getAssessmentColumnSourceType(traces, 'tone')).toBe('HUMAN');
  });

  test('returns undefined when the sources differ', () => {
    const traces = [
      makeTrace('a', { assessments: [makeFeedbackAssessment('tone', 'yes', { source: human })] }),
      makeTrace('b', { assessments: [makeFeedbackAssessment('tone', 'no')] }),
    ];

    expect(getAssessmentColumnSourceType(traces, 'tone')).toBeUndefined();
  });

  test('uses the latest assessment a cell shows, not an older one with another source', () => {
    const traces = [
      makeTrace('a', {
        assessments: [
          makeFeedbackAssessment('tone', 'yes', { create_time: '2025-01-01T00:00:00.000Z' }),
          makeFeedbackAssessment('tone', 'no', { source: human, create_time: '2025-01-02T00:00:00.000Z' }),
        ],
      }),
    ];

    expect(getAssessmentColumnSourceType(traces, 'tone')).toBe('HUMAN');
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

describe('session-level assessment columns', () => {
  const sessionFeedback = (name: string, value: string | number, createTime = '2025-01-01T00:00:00.000Z') =>
    makeFeedbackAssessment(name, value, {
      metadata: { [ASSESSMENT_SESSION_METADATA_KEY]: 'session-1' },
      create_time: createTime,
    });
  const traces = [
    traceWith('turn-1', [makeFeedbackAssessment('correctness', 'yes'), sessionFeedback('completeness', 'yes')]),
    traceWith('turn-2', [makeFeedbackAssessment('correctness', 'no')]),
  ];

  test('are not columns by default', () => {
    expect(computeAssessmentColumns(traces, {}).candidateNames).toEqual(['correctness']);
  });

  test('become columns when session assessments are included', () => {
    expect(computeAssessmentColumns(traces, {}, { includeSessionAssessments: true })).toEqual({
      candidateNames: ['completeness', 'correctness'],
      visibleNames: ['completeness', 'correctness'],
    });
  });

  test('getSessionAssessmentNames lists only valid session-level names', () => {
    const invalid = makeFeedbackAssessment('ignored', 'x', {
      metadata: { [ASSESSMENT_SESSION_METADATA_KEY]: 'session-1' },
      valid: false,
    });
    expect([...getSessionAssessmentNames([...traces, traceWith('turn-3', [invalid])])]).toEqual(['completeness']);
  });

  test('pickSessionCellAssessment finds the latest session value across the turns, and only session values', () => {
    const later = traceWith('turn-3', [sessionFeedback('completeness', 'no', '2025-02-01T00:00:00.000Z')]);
    expect(pickSessionCellAssessment([...traces, later], 'completeness')).toEqual(
      expect.objectContaining({ feedback: { value: 'no' } }),
    );
    expect(pickSessionCellAssessment(traces, 'correctness')).toBeUndefined();
    expect(pickSessionCellAssessment([traces[1]], 'completeness')).toBeUndefined();
  });

  test('getAssessmentColumnType reads session values only when they are included', () => {
    const numeric = [traceWith('turn-1', [sessionFeedback('score', 0.5)])];
    expect(getAssessmentColumnType(numeric, 'score')).toBe('categorical');
    expect(getAssessmentColumnType(numeric, 'score', { includeSessionAssessments: true })).toBe('numeric');
  });
});
