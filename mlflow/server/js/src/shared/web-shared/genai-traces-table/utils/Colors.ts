import { isNil } from 'lodash';

import type { ThemeType } from '@databricks/design-system';

import { KnownEvaluationResultAssessmentStringValue, withAlpha } from '../components/GenAiEvaluationTracesReview.utils';
import type { AssessmentInfo, AssessmentValueType, RunEvaluationResultAssessment } from '../types';

// Taken from figma: https://www.figma.com/design/2B1KMp9x624WrxaASrSv9B/Tiles-UX?node-id=3205-87588&t=1MwrDNNRIOSODm4D-0
export const AGGREGATE_SCORE_CHANGE_BACKGROUND_COLORS = {
  // Tag green
  up: '#02B30214',
  // Tag coral
  down: '#F000400F',
};

// tagTextCoral
export const AGGREGATE_SCORE_CHANGE_TEXT_COLOR = '#64172B';

// From https://www.figma.com/design/HvkTlHxw4sKE77wDlRBDt2/Evaluation-UX?node-id=2996-40835&t=uqVDwh0gqqRJI3jS-0
export const CURRENT_RUN_COLOR = '#077A9D';
export const COMPARE_TO_RUN_COLOR = '#FFAB00';

const PASS_BARCHART_BAR_COLOR = '#99DDB4';
const FAIL_BARCHART_BAR_COLOR = '#FCA4A1';
const ERROR_BARCHART_BAR_COLOR = '#f09065';

const TAG_PASS_COLOR = '#02B30214'; // From tagBackgroundLime.

const BUILTIN_SCORER_ASSESSMENT_COLORS = {
  user_frustration: {
    name: 'user_frustration',
    valueColors: {
      none: {
        bar: PASS_BARCHART_BAR_COLOR,
        tag: TAG_PASS_COLOR,
        icon: (theme: ThemeType) => (theme.isDarkMode ? theme.colors.green400 : theme.colors.green600),
        text: (theme: ThemeType) => (theme.isDarkMode ? theme.colors.green400 : theme.colors.green600),
      },
      resolved: {
        bar: (theme: ThemeType) => theme.colors.yellow400,
        tag: (theme: ThemeType) => (theme.isDarkMode ? withAlpha(theme.colors.yellow800, 0.6) : theme.colors.yellow200),
        icon: (theme: ThemeType) => (theme.isDarkMode ? theme.colors.yellow400 : theme.colors.yellow600),
        text: (theme: ThemeType) => (theme.isDarkMode ? theme.colors.yellow400 : theme.colors.yellow600),
      },
      unresolved: {
        bar: FAIL_BARCHART_BAR_COLOR,
        tag: (theme: ThemeType) => (theme.isDarkMode ? withAlpha(theme.colors.red800, 0.6) : theme.colors.red200),
        icon: (theme: ThemeType) => (theme.isDarkMode ? theme.colors.red400 : theme.colors.red600),
        text: (theme: ThemeType) => (theme.isDarkMode ? theme.colors.red400 : theme.colors.red600),
      },
    },
  },
} as const;

const getBuiltInScorerAssessmentColors = (
  assessmentInfo: AssessmentInfo,
  assessmentValue: AssessmentValueType | string | null | undefined,
  theme: ThemeType,
):
  | {
      bar?: string;
      tag?: string;
      icon?: string;
      text?: string;
    }
  | undefined => {
  const builtInAssessment =
    BUILTIN_SCORER_ASSESSMENT_COLORS[assessmentInfo.name as keyof typeof BUILTIN_SCORER_ASSESSMENT_COLORS];
  if (!builtInAssessment) {
    return undefined;
  }

  const valueKey = String(assessmentValue ?? '') as keyof typeof builtInAssessment.valueColors;
  const colors = builtInAssessment.valueColors[valueKey];
  if (!colors) {
    return undefined;
  }

  return {
    bar: typeof colors.bar === 'function' ? colors.bar(theme) : colors.bar,
    tag: typeof colors.tag === 'function' ? colors.tag(theme) : colors.tag,
    icon: typeof colors.icon === 'function' ? colors.icon(theme) : colors.icon,
    text: typeof colors.text === 'function' ? colors.text(theme) : colors.text,
  };
};

export const getEvaluationResultIconColor = (
  theme: ThemeType,
  assessmentInfo: AssessmentInfo,
  assessment?: RunEvaluationResultAssessment | { stringValue: string; errorMessage?: string },
) => {
  const builtInScorerColors = getBuiltInScorerAssessmentColors(assessmentInfo, assessment?.stringValue, theme);
  if (builtInScorerColors?.icon) {
    return builtInScorerColors.icon;
  }

  if (assessmentInfo.dtype === 'pass-fail') {
    // Return the color based on the assessment value
    if (assessment?.stringValue === KnownEvaluationResultAssessmentStringValue.YES) {
      return theme.isDarkMode ? theme.colors.green400 : theme.colors.green600;
    }
    if (assessment?.stringValue === KnownEvaluationResultAssessmentStringValue.NO) {
      return theme.isDarkMode ? theme.colors.red400 : theme.colors.red600;
    }
  }
  if (assessment?.errorMessage) {
    return theme.colors.textValidationWarning;
  }
  return theme.colors.grey400;
};

export const getEvaluationResultAssessmentBackgroundColor = (
  theme: ThemeType,
  assessmentInfo: AssessmentInfo,
  assessment?: RunEvaluationResultAssessment | { stringValue: string; booleanValue?: string; errorMessage?: string },
  iconOnly = false,
) => {
  const builtInScorerColors = getBuiltInScorerAssessmentColors(assessmentInfo, assessment?.stringValue, theme);
  if (builtInScorerColors?.tag) {
    return builtInScorerColors.tag;
  }

  if (assessmentInfo.dtype === 'pass-fail') {
    // Return the color based on the assessment value
    if (assessment?.stringValue === KnownEvaluationResultAssessmentStringValue.YES) {
      return TAG_PASS_COLOR;
    }
    if (assessment?.stringValue === KnownEvaluationResultAssessmentStringValue.NO) {
      return theme.isDarkMode ? withAlpha(theme.colors.red800, 0.6) : theme.colors.red200;
    }
    if (!iconOnly && assessment?.errorMessage) {
      return '';
    }
  } else if (assessmentInfo.dtype === 'boolean') {
    if (isNil(assessment?.booleanValue)) {
      return '';
    }
    return assessment.booleanValue ? TAG_PASS_COLOR : theme.isDarkMode ? theme.colors.red800 : theme.colors.red200;
  }
  return '';
};

export const getEvaluationResultTextColor = (
  theme: ThemeType,
  assessmentInfo: AssessmentInfo,
  assessment?: RunEvaluationResultAssessment | { stringValue?: string; booleanValue?: boolean; errorMessage?: string },
) => {
  if (assessment?.errorMessage) {
    return theme.colors.textValidationWarning;
  }

  const builtInScorerColors = getBuiltInScorerAssessmentColors(assessmentInfo, assessment?.stringValue, theme);
  if (builtInScorerColors?.text) {
    return builtInScorerColors.text;
  }

  if (assessmentInfo.dtype === 'pass-fail') {
    // Return the color based on the assessment value
    if (assessment?.stringValue === KnownEvaluationResultAssessmentStringValue.YES) {
      return theme.isDarkMode ? theme.colors.green400 : theme.colors.green600;
    } else if (assessment?.stringValue === KnownEvaluationResultAssessmentStringValue.NO) {
      return theme.isDarkMode ? theme.colors.red400 : theme.colors.red600;
    } else {
      return theme.colors.textSecondary;
    }
  } else if (assessmentInfo.dtype === 'boolean') {
    if (isNil(assessment?.booleanValue)) {
      return theme.colors.textSecondary;
    }
    return theme.colors.textPrimary;
  } else if (assessmentInfo.dtype === 'unknown') {
    return theme.colors.textSecondary;
  }
  return theme.colors.textPrimary;
};

export const getAssessmentValueBarBackgroundColor = (
  theme: ThemeType,
  assessmentInfo: AssessmentInfo,
  assessmentValue: AssessmentValueType,
  isError?: boolean,
) => {
  if (isError) {
    return ERROR_BARCHART_BAR_COLOR;
  }

  const builtInScorerColors = getBuiltInScorerAssessmentColors(assessmentInfo, assessmentValue, theme);
  if (builtInScorerColors?.bar) {
    return builtInScorerColors.bar;
  }

  if (assessmentInfo.dtype === 'pass-fail') {
    // Return the color based on the assessment value
    if (assessmentValue === KnownEvaluationResultAssessmentStringValue.YES) {
      return PASS_BARCHART_BAR_COLOR;
    }
    if (assessmentValue === KnownEvaluationResultAssessmentStringValue.NO) {
      return FAIL_BARCHART_BAR_COLOR;
    }
    return theme.isDarkMode ? theme.colors.grey800 : theme.colors.grey200;
  } else if (assessmentInfo.dtype === 'boolean') {
    if (isNil(assessmentValue)) {
      return theme.isDarkMode ? theme.colors.grey800 : theme.colors.grey200;
    }
    return assessmentValue ? PASS_BARCHART_BAR_COLOR : FAIL_BARCHART_BAR_COLOR;
  }
  return theme.isDarkMode ? theme.colors.grey800 : theme.colors.grey200;
};
