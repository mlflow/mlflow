import type { Assessment } from '../ModelTrace.types';

export const getAssessmentValue = (assessment: Assessment) => {
  if ('feedback' in assessment) {
    return assessment.feedback.value;
  }

  if ('value' in assessment.expectation) {
    return assessment.expectation.value;
  }

  return assessment.expectation.serialized_value.value;
};
