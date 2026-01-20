import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';

import { isAssessmentPassing } from '../components/EvaluationsReviewAssessmentTag';
import type { AssessmentInfo } from '../types';

/**
 * Result of aggregating assessment values across multiple traces
 */
export interface AssessmentAggregationResult {
  passCount: number;
  totalCount: number;
}

/**
 * Aggregates assessment pass/fail counts across multiple traces for a given assessment.
 * A trace can have multiple assessments with the same name, and they all count toward the total.
 *
 * @param traces - Array of trace info objects to aggregate
 * @param assessmentInfo - The assessment metadata including name and dtype
 * @returns Object containing passCount and totalCount
 */
export function aggregatePassFailAssessmentsFromTraces(
  traces: ModelTraceInfoV3[],
  assessmentInfo: AssessmentInfo,
): AssessmentAggregationResult {
  let passCount = 0;
  let totalCount = 0;

  for (const trace of traces) {
    const assessments =
      trace.assessments?.filter(
        (assessment) => assessment.valid !== false && assessment.assessment_name === assessmentInfo.name,
      ) ?? [];
    for (const assessment of assessments) {
      if (!('feedback' in assessment)) {
        continue;
      }

      const feedbackValue = assessment.feedback.value;
      const isPassing = isAssessmentPassing(assessmentInfo, feedbackValue);

      if (isPassing === undefined) {
        continue;
      }

      totalCount++;
      if (isPassing) {
        passCount++;
      }
    }
  }

  return { passCount, totalCount };
}
