import { describe, it, expect } from '@jest/globals';
import type { ThemeType } from '@databricks/design-system';
import {
  getEvaluationResultIconColor,
  getEvaluationResultAssessmentBackgroundColor,
  getEvaluationResultTextColor,
  getAssessmentValueBarBackgroundColor,
  PASS_BARCHART_BAR_COLOR,
  FAIL_BARCHART_BAR_COLOR,
} from './Colors';
import type { AssessmentInfo } from '../types';

describe('Colors utils - PASS/FAIL support', () => {
  const mockTheme = {
    isDarkMode: false,
    colors: {
      green600: 'mock-green-600',
      red600: 'mock-red-600',
      red200: 'mock-red-200',
      grey400: 'mock-grey-400',
      grey200: 'mock-grey-200',
      textSecondary: 'mock-text-secondary',
    },
  } as unknown as ThemeType;

  const mockAssessmentInfo = {
    name: 'test_metric',
    dtype: 'pass-fail',
  } as AssessmentInfo;

  describe('getEvaluationResultIconColor', () => {
    it('should return green for YES and PASS', () => {
      expect(getEvaluationResultIconColor(mockTheme, mockAssessmentInfo, { stringValue: 'yes' })).toBe(
        'mock-green-600',
      );
      expect(getEvaluationResultIconColor(mockTheme, mockAssessmentInfo, { stringValue: 'PASS' })).toBe(
        'mock-green-600',
      );
    });

    it('should handle case-insensitive PASS/FAIL values', () => {
      expect(getEvaluationResultIconColor(mockTheme, mockAssessmentInfo, { stringValue: 'pass' })).toBe(
        'mock-green-600',
      );
      expect(getEvaluationResultIconColor(mockTheme, mockAssessmentInfo, { stringValue: 'fAiL' })).toBe('mock-red-600');
    });

    it('should return red for NO and FAIL', () => {
      expect(getEvaluationResultIconColor(mockTheme, mockAssessmentInfo, { stringValue: 'no' })).toBe('mock-red-600');
      expect(getEvaluationResultIconColor(mockTheme, mockAssessmentInfo, { stringValue: 'FAIL' })).toBe('mock-red-600');
    });
  });

  describe('getEvaluationResultAssessmentBackgroundColor', () => {
    const TAG_PASS_COLOR = '#02B30214';

    it('should return PASS background color for YES and PASS', () => {
      expect(getEvaluationResultAssessmentBackgroundColor(mockTheme, mockAssessmentInfo, { stringValue: 'yes' })).toBe(
        TAG_PASS_COLOR,
      );
      expect(getEvaluationResultAssessmentBackgroundColor(mockTheme, mockAssessmentInfo, { stringValue: 'PASS' })).toBe(
        TAG_PASS_COLOR,
      );
    });

    it('should return FAIL background color for NO and FAIL', () => {
      expect(getEvaluationResultAssessmentBackgroundColor(mockTheme, mockAssessmentInfo, { stringValue: 'no' })).toBe(
        'mock-red-200',
      );
      expect(getEvaluationResultAssessmentBackgroundColor(mockTheme, mockAssessmentInfo, { stringValue: 'FAIL' })).toBe(
        'mock-red-200',
      );
    });

    it('should handle case-insensitive PASS/FAIL values', () => {
      expect(getEvaluationResultAssessmentBackgroundColor(mockTheme, mockAssessmentInfo, { stringValue: 'PaSs' })).toBe(
        TAG_PASS_COLOR,
      );
      expect(getEvaluationResultAssessmentBackgroundColor(mockTheme, mockAssessmentInfo, { stringValue: 'fail' })).toBe(
        'mock-red-200',
      );
    });
  });

  describe('getEvaluationResultTextColor', () => {
    it('should return green text for YES and PASS', () => {
      expect(getEvaluationResultTextColor(mockTheme, mockAssessmentInfo, { stringValue: 'yes' })).toBe(
        'mock-green-600',
      );
      expect(getEvaluationResultTextColor(mockTheme, mockAssessmentInfo, { stringValue: 'PASS' })).toBe(
        'mock-green-600',
      );
    });

    it('should return red text for NO and FAIL', () => {
      expect(getEvaluationResultTextColor(mockTheme, mockAssessmentInfo, { stringValue: 'no' })).toBe('mock-red-600');
      expect(getEvaluationResultTextColor(mockTheme, mockAssessmentInfo, { stringValue: 'FAIL' })).toBe('mock-red-600');
    });

    it('should handle case-insensitive PASS/FAIL values', () => {
      expect(getEvaluationResultTextColor(mockTheme, mockAssessmentInfo, { stringValue: 'pass' })).toBe(
        'mock-green-600',
      );
      expect(getEvaluationResultTextColor(mockTheme, mockAssessmentInfo, { stringValue: 'FaIl' })).toBe('mock-red-600');
    });
  });

  describe('getAssessmentValueBarBackgroundColor', () => {
    it('should return PASS bar chart color for YES and PASS', () => {
      expect(getAssessmentValueBarBackgroundColor(mockTheme, mockAssessmentInfo, 'yes')).toBe(PASS_BARCHART_BAR_COLOR);
      expect(getAssessmentValueBarBackgroundColor(mockTheme, mockAssessmentInfo, 'PASS')).toBe(PASS_BARCHART_BAR_COLOR);
    });

    it('should return FAIL bar chart color for NO and FAIL', () => {
      expect(getAssessmentValueBarBackgroundColor(mockTheme, mockAssessmentInfo, 'no')).toBe(FAIL_BARCHART_BAR_COLOR);
      expect(getAssessmentValueBarBackgroundColor(mockTheme, mockAssessmentInfo, 'FAIL')).toBe(FAIL_BARCHART_BAR_COLOR);
    });

    it('should handle case-insensitive PASS/FAIL values', () => {
      expect(getAssessmentValueBarBackgroundColor(mockTheme, mockAssessmentInfo, 'pAsS')).toBe(PASS_BARCHART_BAR_COLOR);
      expect(getAssessmentValueBarBackgroundColor(mockTheme, mockAssessmentInfo, 'fail')).toBe(FAIL_BARCHART_BAR_COLOR);
    });
  });
});
