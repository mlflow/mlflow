import { SparkleIcon, UserIcon, CodeIcon } from '@databricks/design-system';

import type { Assessment, ExpectationAssessment, FeedbackAssessment } from '../ModelTrace.types';

export const getAssessmentValue = (assessment: Assessment) => {
  if ('feedback' in assessment && assessment.feedback) {
    return assessment.feedback.value;
  }

  if ('expectation' in assessment) {
    if (assessment.expectation && 'value' in assessment.expectation) {
      return assessment.expectation.value;
    }
    return assessment.expectation.serialized_value.value;
  }

  return undefined;
};

export const isFeedbackAssessment = (assessment: Assessment): assessment is FeedbackAssessment => {
  return 'feedback' in assessment && Boolean(assessment.feedback);
};

export const isExpectationAssessment = (assessment: Assessment): assessment is ExpectationAssessment => {
  return 'expectation' in assessment && Boolean(assessment.expectation);
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
