import { describe, expect, test } from '@jest/globals';
import { makeFeedbackAssessment, makeTrace } from '../test-utils/mockTraces';
import { assessmentValueKindKey, classifyAssessmentValueKinds } from './assessmentColumns';

describe('classifyAssessmentValueKinds', () => {
  test('numeric only when every loaded value is a number', () => {
    const kinds = classifyAssessmentValueKinds([
      makeTrace('t1', {
        assessments: [
          makeFeedbackAssessment('relevance', 4),
          makeFeedbackAssessment('rating', '5'),
          makeFeedbackAssessment('mixed', 3),
          makeFeedbackAssessment('safety', true),
        ],
      }),
      makeTrace('t2', {
        assessments: [makeFeedbackAssessment('relevance', 2.5), makeFeedbackAssessment('mixed', 'n/a')],
      }),
    ]);

    expect(kinds.get(assessmentValueKindKey('feedback', 'relevance'))).toBe('numeric');
    expect(kinds.get(assessmentValueKindKey('feedback', 'rating'))).toBe('other');
    expect(kinds.get(assessmentValueKindKey('feedback', 'mixed'))).toBe('other');
    expect(kinds.get(assessmentValueKindKey('feedback', 'safety'))).toBe('other');
    expect(kinds.has(assessmentValueKindKey('expectation', 'relevance'))).toBe(false);
  });

  test('an errored or empty value does not turn numeric matching off', () => {
    const kinds = classifyAssessmentValueKinds([
      makeTrace('t1', { assessments: [makeFeedbackAssessment('relevance', 4)] }),
      makeTrace('t2', {
        assessments: [
          makeFeedbackAssessment('relevance', 0, {
            feedback: { value: null, error: { error_code: 'SCORER_ERROR', error_message: 'timed out' } },
          } as Parameters<typeof makeFeedbackAssessment>[2]),
        ],
      }),
    ]);

    expect(kinds.get(assessmentValueKindKey('feedback', 'relevance'))).toBe('numeric');
  });

  test('ignores assessments that get no column', () => {
    const kinds = classifyAssessmentValueKinds([
      makeTrace('t1', { assessments: [makeFeedbackAssessment('stale', 1, { valid: false })] }),
    ]);
    expect(kinds.size).toBe(0);
  });
});
