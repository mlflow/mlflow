import { describe, expect, jest, test } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';
import { shouldEnableNewEvalRunsUi } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { makeFeedbackAssessment, makeTrace } from '../test-utils/mockTraces';
import { assessmentValueKindKey } from '../utils/assessmentColumns';
import { useNumericAssessmentKeys } from './useNumericAssessmentKeys';

jest.mock('@mlflow/mlflow/src/common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('@mlflow/mlflow/src/common/utils/FeatureUtils')>(
    '@mlflow/mlflow/src/common/utils/FeatureUtils',
  ),
  shouldEnableNewEvalRunsUi: jest.fn(() => true),
}));

const RELEVANCE = assessmentValueKindKey('feedback', 'relevance');
const page = (value: number | string) => [
  makeTrace('t', { assessments: [makeFeedbackAssessment('relevance', value)] }),
];

describe('useNumericAssessmentKeys', () => {
  test('remembers numeric names across pages, and text wins over numbers', () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    const { result } = renderHook(() => useNumericAssessmentKeys('exp-1'));
    expect(result.current.numericAssessmentKeys.size).toBe(0);

    act(() => result.current.recordTraces(page(4)));
    expect(result.current.numericAssessmentKeys.has(RELEVANCE)).toBe(true);

    // A page without the assessment (such as a filtered page with no rows) keeps what was learned.
    act(() => result.current.recordTraces([]));
    expect(result.current.numericAssessmentKeys.has(RELEVANCE)).toBe(true);

    act(() => result.current.recordTraces(page('n/a')));
    expect(result.current.numericAssessmentKeys.has(RELEVANCE)).toBe(false);
    act(() => result.current.recordTraces(page(5)));
    expect(result.current.numericAssessmentKeys.has(RELEVANCE)).toBe(false);
  });

  test('starts over for a new scope', () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    const { result, rerender } = renderHook(({ scope }) => useNumericAssessmentKeys(scope), {
      initialProps: { scope: 'exp-1' },
    });
    act(() => result.current.recordTraces(page(4)));
    rerender({ scope: 'exp-2' });
    expect(result.current.numericAssessmentKeys.size).toBe(0);
  });

  test('records nothing with the new evaluation runs UI off', () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
    const { result } = renderHook(() => useNumericAssessmentKeys('exp-1'));
    const before = result.current.numericAssessmentKeys;
    act(() => result.current.recordTraces(page(4)));
    expect(result.current.numericAssessmentKeys).toBe(before);
    expect(before.size).toBe(0);
  });
});
