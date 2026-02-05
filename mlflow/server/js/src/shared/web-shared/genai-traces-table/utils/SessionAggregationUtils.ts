import { isSessionLevelAssessment, type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';

import { isAssessmentPassing } from '../components/EvaluationsReviewAssessmentTag';
import { ASSESSMENT_SESSION_METADATA_KEY } from '../../model-trace-explorer/constants';
import type { AssessmentInfo } from '../types';

/**
 * Result of aggregating pass/fail assessment values across multiple traces
 */
export interface PassFailAggregationResult {
  passCount: number;
  totalCount: number;
}

/**
 * Result of aggregating numeric assessment values across multiple traces
 */
export interface NumericAggregationResult {
  average: number | null;
  count: number;
}

/**
 * Result of aggregating string assessment values across multiple traces
 */
export interface StringAggregationResult {
  valueCounts: Map<string, number>;
  totalCount: number;
}

/**
 * Filters assessments from a trace by name, excluding session-level and invalid assessments.
 */
function getValidAssessments(trace: ModelTraceInfoV3, assessmentName: string) {
  return (
    trace.assessments?.filter(
      (a) => a.assessment_name === assessmentName && !isSessionLevelAssessment(a) && a.valid !== false,
    ) ?? []
  );
}

/**
 * Aggregates assessment pass/fail counts across multiple traces for a given assessment.
 * A trace can have multiple assessments with the same name, and they all count toward the total.
 *
 * @param traces - Array of trace info objects to aggregate
 * @param assessmentInfo - The assessment metadata including name and dtype
 * @returns Object containing passCount and totalCount
 */
export function aggregatePassFailAssessments(
  traces: ModelTraceInfoV3[],
  assessmentInfo: AssessmentInfo,
): PassFailAggregationResult {
  let passCount = 0;
  let totalCount = 0;

  for (const trace of traces) {
    const assessments = getValidAssessments(trace, assessmentInfo.name);

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

/**
 * Aggregates numeric assessment values across multiple traces.
 * Returns the average and count of all numeric values.
 *
 * @param traces - Array of trace info objects to aggregate
 * @param assessmentName - The name of the assessment to aggregate
 * @returns Object containing average (or null if no values) and count
 */
export function aggregateNumericAssessments(
  traces: ModelTraceInfoV3[],
  assessmentName: string,
): NumericAggregationResult {
  let sum = 0;
  let count = 0;

  for (const trace of traces) {
    const assessments = getValidAssessments(trace, assessmentName);

    for (const assessment of assessments) {
      if (!('feedback' in assessment)) {
        continue;
      }

      const feedbackValue = assessment.feedback.value;
      if (typeof feedbackValue === 'number' && !isNaN(feedbackValue)) {
        sum += feedbackValue;
        count++;
      }
    }
  }

  return {
    average: count > 0 ? sum / count : null,
    count,
  };
}

/**
 * Aggregates string assessment values across multiple traces.
 * Returns unique values with their counts.
 *
 * @param traces - Array of trace info objects to aggregate
 * @param assessmentName - The name of the assessment to aggregate
 * @returns Object containing valueCounts map and totalCount
 */
export function aggregateStringAssessments(
  traces: ModelTraceInfoV3[],
  assessmentName: string,
): StringAggregationResult {
  const valueCounts = new Map<string, number>();
  let totalCount = 0;

  for (const trace of traces) {
    const assessments = getValidAssessments(trace, assessmentName);

    for (const assessment of assessments) {
      if (!('feedback' in assessment)) {
        continue;
      }

      const feedbackValue = assessment.feedback.value;
      if (typeof feedbackValue === 'string') {
        valueCounts.set(feedbackValue, (valueCounts.get(feedbackValue) ?? 0) + 1);
        totalCount++;
      }
    }
  }

  return { valueCounts, totalCount };
}
