import { describe, it, expect } from '@jest/globals';
import { isAssessmentPassing } from './EvaluationsReviewAssessmentTag';
import type { AssessmentInfo } from '../types';

describe('isAssessmentPassing', () => {
  const mockPassFailAssessment = { dtype: 'pass-fail' } as AssessmentInfo;
  const mockBooleanAssessment = { dtype: 'boolean' } as AssessmentInfo;

  it('should return true for YES and PASS values', () => {
    expect(isAssessmentPassing(mockPassFailAssessment, 'yes')).toBe(true);
    expect(isAssessmentPassing(mockPassFailAssessment, 'PASS')).toBe(true);
  });

  it('should return false for NO and FAIL values', () => {
    expect(isAssessmentPassing(mockPassFailAssessment, 'no')).toBe(false);
    expect(isAssessmentPassing(mockPassFailAssessment, 'FAIL')).toBe(false);
  });

  it('should return undefined for unrecognized values in pass-fail', () => {
    expect(isAssessmentPassing(mockPassFailAssessment, 'maybe')).toBeUndefined();
    expect(isAssessmentPassing(mockPassFailAssessment, null)).toBeUndefined();
  });

  it('should handle boolean dtypes correctly', () => {
    expect(isAssessmentPassing(mockBooleanAssessment, true)).toBe(true);
    expect(isAssessmentPassing(mockBooleanAssessment, false)).toBe(false);
  });
});
