import type { Assessment } from '../ModelTrace.types';
import { useDesignSystemTheme, Typography, SparkleIcon, UserIcon, CodeIcon } from '@databricks/design-system';

export const getAssessmentValue = (assessment: Assessment) => {
  if ('feedback' in assessment) {
    return assessment.feedback.value;
  }

  if ('value' in assessment.expectation) {
    return assessment.expectation.value;
  }

  return assessment.expectation.serialized_value.value;
};

export const getSourceIcon = (source: Assessment['source']) => {
  switch (source.source_type) {
    case 'HUMAN':
      return UserIcon;
    case 'LLM_JUDGE':
      return SparkleIcon;
    default:
      return CodeIcon;
  }
};
