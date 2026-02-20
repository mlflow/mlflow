/**
 * Utilities for calculating labeling progress and completion status
 */

import type { ThemeType } from '@databricks/design-system';

import type { ModelTraceInfoV3 } from '../../../shared/web-shared/model-trace-explorer';
import type {
  LabelingItem,
  LabelingSchema,
  LabelingSession,
} from '../../types/labeling';
import { getValidAssessmentValue } from '../../types/labeling';
import type { Assessment } from '../../../shared/web-shared/model-trace-explorer/ModelTrace.types';

/**
 * Trace type with assessments - simplified for labeling
 */
export interface LabelingTrace {
  traceInfo: ModelTraceInfoV3;
  assessments?: Assessment[];
}

/**
 * Given a trace, for each labeling schema in the session, check if the task is considered completed.
 */
export function getLabelingProgressForTrace({
  session,
  trace,
}: {
  session: LabelingSession;
  trace: LabelingTrace;
}): { schemaName: string; completed: boolean }[] {
  return session.labelingSchemas.map((schema) => {
    const assessment = getAssessmentForLabelingSchema(trace, schema);
    return {
      schemaName: schema.name,
      completed: isTaskCompleted(assessment, schema),
    };
  });
}

/**
 * Type guards for Assessment types
 */
function isFeedbackAssessment(
  assessment: Assessment,
): assessment is import('../../../shared/web-shared/model-trace-explorer/ModelTrace.types').FeedbackAssessment {
  return 'feedback' in assessment;
}

/**
 * Given a trace, return the assessment for the labeling schema.
 *
 * For OSS MLflow, we simply find any assessment matching the schema name and type.
 * In Databricks, this would be filtered by current user.
 */
export function getAssessmentForLabelingSchema(
  trace: LabelingTrace,
  schema: LabelingSchema,
): Assessment | undefined {
  return trace.assessments?.find((assessment) => {
    // Check if the assessment name matches
    if (assessment.assessment_name !== schema.name) {
      return false;
    }

    // Check if the assessment type matches the schema type
    const isFeedback = isFeedbackAssessment(assessment);
    const assessmentType = isFeedback ? 'FEEDBACK' : 'EXPECTATION';
    return assessmentType === schema.type;
  });
}

/**
 * Check if the task is considered completed for the labeling schema based on the assessment.
 */
export function isTaskCompleted(
  assessment: Assessment | undefined,
  schema: LabelingSchema,
): boolean {
  const validAssessmentValue = getValidAssessmentValue(assessment, schema);

  // Not complete if assessment does not have a valid value for the schema
  if (validAssessmentValue == null) {
    return false;
  }

  // For now, we don't enforce any additional completion criteria for the schema
  // but we can add more rules here if needed.
  return true;
}

/**
 * Overall review progress for all labeling items.
 */
export function computeOverallReviewProgress(
  theme: ThemeType,
  labelingItems: LabelingItem[] | undefined,
  maxNumProgressBarItems: number = 100,
) {
  const totalCount = labelingItems?.length ?? 0;
  const reviewedCount = labelingItems?.filter((item) => item.state === 'COMPLETED').length ?? 0;
  const percentage = totalCount === 0 ? 0 : Math.round((reviewedCount / totalCount) * 100);

  const progressBarColor = (state: string) => {
    if (state === 'COMPLETED') {
      return theme.colors.blue600;
    } else {
      return theme.isDarkMode ? theme.colors.blue800 : theme.colors.blue200;
    }
  };

  const progressBarItems =
    totalCount < maxNumProgressBarItems
      ? labelingItems?.map((_e, index) => ({
          color: progressBarColor(index >= reviewedCount ? 'IN_PROGRESS' : 'COMPLETED'),
        })) ?? []
      : Array.from({ length: maxNumProgressBarItems }, (_, index) => ({
          color: progressBarColor(
            Math.round((index / maxNumProgressBarItems) * 100) >= percentage
              ? 'IN_PROGRESS'
              : 'COMPLETED',
          ),
        }));

  return {
    progressBarItems,
    percentage,
    totalCount,
    reviewedCount,
  };
}
