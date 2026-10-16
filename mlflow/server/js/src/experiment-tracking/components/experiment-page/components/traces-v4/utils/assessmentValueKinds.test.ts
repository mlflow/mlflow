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

  test('ignores assessments that get no column', () => {
    const kinds = classifyAssessmentValueKinds([
      makeTrace('t1', { assessments: [makeFeedbackAssessment('stale', 1, { valid: false })] }),
    ]);
    expect(kinds.size).toBe(0);
  });
});
