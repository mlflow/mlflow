import type { Assessment, Expectation, Feedback } from './ModelTrace.types';
import { fetchAPI } from './ModelTraceExplorer.request.utils';

export const deleteAssessment = ({ traceId, assessmentId }: { traceId: string; assessmentId: string }) =>
  fetchAPI(`/ajax-api/3.0/mlflow/traces/${traceId}/assessments/${assessmentId}`, 'DELETE');

// these fields are set by the server on create
export type CreateAssessmentPayload = {
  assessment: Omit<Assessment, 'assessment_id' | 'create_time' | 'last_update_time'>;
};

export const createAssessment = ({ payload }: { payload: CreateAssessmentPayload }) =>
  fetchAPI(`/ajax-api/3.0/mlflow/traces/${payload.assessment.trace_id}/assessments`, 'POST', payload);

export const fetchTraceInfoV3 = ({ traceId }: { traceId: string }) =>
  fetchAPI(`/ajax-api/3.0/mlflow/traces/${traceId}`);

export type UpdateAssessmentPayload = {
  // we only support updating these fields
  assessment: {
    feedback?: Feedback;
    expectation?: Expectation;
    rationale?: string;
    metadata?: Record<string, string>;
  };
  // comma-separated list of fields to update
  update_mask: string;
};

export const updateAssessment = ({
  traceId,
  assessmentId,
  payload,
}: {
  traceId: string;
  assessmentId: string;
  payload: UpdateAssessmentPayload;
}) => fetchAPI(`/ajax-api/3.0/mlflow/traces/${traceId}/assessments/${assessmentId}`, 'PATCH', payload);
