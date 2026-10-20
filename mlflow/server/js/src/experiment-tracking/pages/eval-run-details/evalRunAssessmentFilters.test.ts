import { describe, expect, test } from '@jest/globals';
import { FilterOp } from '@databricks/web-shared/traces-table';
import type { ExpectationAssessment } from '@databricks/web-shared/model-trace-explorer';
import {
  makeFeedbackAssessment,
  makeTrace,
} from '../../components/experiment-page/components/traces-v4/test-utils/mockTraces';
import { matchesEvalRunAssessmentFilter, splitEvalRunAssessmentFilters } from './evalRunAssessmentFilters';

const feedbackClause = (key: string, value: string, operator = FilterOp.EQUALS) => ({
  field: 'assessment',
  key,
  operator,
  value,
});

const makeExpectation = (name: string, value: string): ExpectationAssessment =>
  ({
    assessment_id: `exp-${name}`,
    assessment_name: name,
    trace_id: 'tr-1',
    source: { source_type: 'HUMAN', source_id: 'user' },
    create_time: '2025-01-01T00:00:00.000Z',
    last_update_time: '2025-01-01T00:00:00.000Z',
    expectation: { value },
  }) as ExpectationAssessment;

describe('splitEvalRunAssessmentFilters', () => {
  test('keeps feedback and expectation clauses for the page and sends the rest to the server', () => {
    const state = { field: 'status', operator: FilterOp.EQUALS, value: 'OK' };
    const feedback = feedbackClause('correctness', 'no');
    const expectation = { field: 'expectation', key: 'answer', operator: FilterOp.EQUALS, value: 'Paris' };
    const incomplete = { field: 'assessment', key: 'correctness', operator: FilterOp.EQUALS, value: '' };

    expect(splitEvalRunAssessmentFilters([state, feedback, expectation, incomplete])).toEqual({
      serverFilterModel: [state, incomplete],
      assessmentFilters: [feedback, expectation],
    });
  });
});

describe('matchesEvalRunAssessmentFilter', () => {
  test("matches the run's own feedback, not another run's on a linked trace", () => {
    // After run scoping, a trace linked to runs A and B keeps only B's "yes".
    const runScopedTrace = makeTrace('tr-1', { assessments: [makeFeedbackAssessment('correctness', 'yes')] });

    expect(matchesEvalRunAssessmentFilter(runScopedTrace, feedbackClause('correctness', 'no'))).toBe(false);
    expect(matchesEvalRunAssessmentFilter(runScopedTrace, feedbackClause('correctness', 'yes'))).toBe(true);
  });

  test('compares numeric values as numbers, on a single assessment', () => {
    const trace = makeTrace('tr-1', { assessments: [makeFeedbackAssessment('relevance', 4)] });

    expect(matchesEvalRunAssessmentFilter(trace, feedbackClause('relevance', '4'))).toBe(true);
    expect(matchesEvalRunAssessmentFilter(trace, feedbackClause('relevance', '4.0'))).toBe(true);
    expect(matchesEvalRunAssessmentFilter(trace, feedbackClause('relevance', '3'))).toBe(false);
    expect(matchesEvalRunAssessmentFilter(trace, feedbackClause('relevance', '4', FilterOp.NOT_EQUALS))).toBe(false);
    expect(matchesEvalRunAssessmentFilter(trace, feedbackClause('relevance', '3', FilterOp.NOT_EQUALS))).toBe(true);
  });

  test('uses the latest assessment with the name', () => {
    const trace = makeTrace('tr-1', {
      assessments: [
        makeFeedbackAssessment('relevance', 3, { create_time: '2025-01-01T00:00:00.000Z' }),
        makeFeedbackAssessment('relevance', 5, { create_time: '2025-01-02T00:00:00.000Z' }),
      ],
    });

    expect(matchesEvalRunAssessmentFilter(trace, feedbackClause('relevance', '5'))).toBe(true);
    expect(matchesEvalRunAssessmentFilter(trace, feedbackClause('relevance', '3'))).toBe(false);
    expect(matchesEvalRunAssessmentFilter(trace, feedbackClause('relevance', '4'))).toBe(false);
  });

  test('keeps feedback and expectations with the same name apart', () => {
    const trace = makeTrace('tr-1', {
      assessments: [makeFeedbackAssessment('answer', 'London'), makeExpectation('answer', 'Paris')],
    });

    expect(matchesEvalRunAssessmentFilter(trace, feedbackClause('answer', 'London'))).toBe(true);
    expect(
      matchesEvalRunAssessmentFilter(trace, {
        field: 'expectation',
        key: 'answer',
        operator: FilterOp.EQUALS,
        value: 'Paris',
      }),
    ).toBe(true);
    expect(matchesEvalRunAssessmentFilter(trace, feedbackClause('answer', 'Paris'))).toBe(false);
  });

  test('a trace without the assessment matches neither = nor !=', () => {
    const trace = makeTrace('tr-1');

    expect(matchesEvalRunAssessmentFilter(trace, feedbackClause('correctness', 'yes'))).toBe(false);
    expect(matchesEvalRunAssessmentFilter(trace, feedbackClause('correctness', 'yes', FilterOp.NOT_EQUALS))).toBe(
      false,
    );
  });
});
