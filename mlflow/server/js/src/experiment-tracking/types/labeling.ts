/**
 * Labeling types for MLflow labeling sessions and schemas
 */

import type { Assessment } from '../../shared/web-shared/model-trace-explorer/ModelTrace.types';

/**
 * Labeling item status
 */
export type LabelingItemStatus = 'PENDING' | 'IN_PROGRESS' | 'COMPLETED' | 'SKIPPED';

/**
 * Assessment type for labeling schemas
 */
export type LabelingAssessmentType = 'FEEDBACK' | 'EXPECTATION';

/**
 * Value type configuration for labeling schemas
 */
export type LabelingSchemaValueType =
  | { type: 'TEXT'; maxLength?: number }
  | { type: 'NUMERIC'; min?: number; max?: number }
  | { type: 'CATEGORICAL'; options: string[] }
  | { type: 'CATEGORICAL_LIST'; options: string[] }
  | { type: 'TEXT_LIST'; maxLength?: number };

/**
 * Labeling schema definition
 */
export interface LabelingSchema {
  labeling_schema_id: string;
  labeling_session_id: string;
  name: string;
  type: LabelingAssessmentType;
  schema: LabelingSchemaValueType;
  title: string;
  instructions?: string;
  creation_time: number;
  last_update_time: number;
}

/**
 * Labeling session
 */
export interface LabelingSession {
  labeling_session_id: string;
  experiment_id: string;
  name: string;
  labelingSchemas: LabelingSchema[];
  creation_time: number;
  last_update_time: number;
}

/**
 * Labeling item
 */
export interface LabelingItem {
  labeling_item_id: string;
  labeling_session_id: string;
  trace_id?: string;
  dataset_record_id?: string;
  dataset_id?: string;
  state: LabelingItemStatus;
  creation_time: number;
  last_update_time: number;
}

/**
 * Type guards for Assessment types
 */
function isFeedbackAssessment(
  assessment: Assessment,
): assessment is import('../../shared/web-shared/model-trace-explorer/ModelTrace.types').FeedbackAssessment {
  return 'feedback' in assessment;
}

function isExpectationAssessment(
  assessment: Assessment,
): assessment is import('../../shared/web-shared/model-trace-explorer/ModelTrace.types').ExpectationAssessment {
  return 'expectation' in assessment;
}

/**
 * Extract the value from an Assessment (handles both Feedback and Expectation)
 */
function getAssessmentValue(
  assessment: Assessment,
): string | number | boolean | string[] | null | undefined {
  if (isFeedbackAssessment(assessment)) {
    return assessment.feedback.value;
  } else if (isExpectationAssessment(assessment)) {
    const expectation = assessment.expectation;
    if ('value' in expectation) {
      return expectation.value;
    }
    // Handle serialized value if needed in the future
    return undefined;
  }
  return undefined;
}

/**
 * Type guard to check if an assessment value is valid for a given schema
 */
export function getValidAssessmentValue(
  assessment: Assessment | undefined,
  schema: LabelingSchema,
): string | number | boolean | string[] | null | undefined {
  if (!assessment) {
    return undefined;
  }

  const value = getAssessmentValue(assessment);
  if (value == null) {
    return undefined;
  }

  // Check assessment type matches schema type
  const isFeedback = isFeedbackAssessment(assessment);
  const assessmentType = isFeedback ? 'FEEDBACK' : 'EXPECTATION';
  if (assessmentType !== schema.type) {
    return undefined;
  }

  // Extract value based on schema value type
  switch (schema.schema.type) {
    case 'TEXT':
    case 'TEXT_LIST':
      return typeof value === 'string' ? value : undefined;
    case 'NUMERIC':
      return typeof value === 'number' ? value : undefined;
    case 'CATEGORICAL':
      if (typeof value === 'string') {
        return (schema.schema as Extract<LabelingSchemaValueType, { type: 'CATEGORICAL' }>).options.includes(
          value,
        )
          ? value
          : undefined;
      }
      return undefined;
    case 'CATEGORICAL_LIST':
      if (Array.isArray(value)) {
        const options = (schema.schema as Extract<LabelingSchemaValueType, { type: 'CATEGORICAL_LIST' }>).options;
        const allValid = value.every(
          (v: unknown) => typeof v === 'string' && options.includes(v),
        );
        return allValid ? value : undefined;
      }
      return undefined;
    default:
      return undefined;
  }
}
