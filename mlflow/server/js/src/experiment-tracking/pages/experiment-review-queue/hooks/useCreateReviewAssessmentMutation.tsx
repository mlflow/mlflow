import { useMutation } from '@databricks/web-shared/query-client';

import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';

export type AssessmentValue = string | number | boolean | string[] | null;

export interface CreateReviewAssessmentParams {
  traceId: string;
  /** The label-schema name; becomes the assessment name. */
  name: string;
  /** Schema kind — decides whether a feedback or expectation is written. */
  assessmentKind: 'feedback' | 'expectation';
  value: AssessmentValue;
  /** Reviewer identifier; `default` on the no-auth surface. */
  sourceId: string;
}

/**
 * Write a reviewer's answer to a queue question as a trace assessment.
 *
 * A `FEEDBACK` schema produces a `feedback` payload (a judgment of the
 * trace); an `EXPECTATION` schema produces an `expectation` payload (a
 * ground-truth label). The `source` is always `HUMAN`. Posts directly to
 * the existing trace-assessments endpoint via the local `fetchAPI` rather
 * than the trace-explorer's `useCreateAssessment` hook, which is coupled to
 * the explorer's provider context.
 */
export const useCreateReviewAssessmentMutation = () => {
  const { mutate, mutateAsync, isLoading, error } = useMutation<unknown, Error, CreateReviewAssessmentParams>({
    mutationFn: async ({ traceId, name, assessmentKind, value, sourceId }) => {
      const assessment = {
        assessment_name: name,
        trace_id: traceId,
        source: { source_type: 'HUMAN', source_id: sourceId },
        ...(assessmentKind === 'feedback' ? { feedback: { value } } : { expectation: { value } }),
      };
      return fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/traces/${traceId}/assessments`), {
        method: 'POST',
        body: { assessment },
      });
    },
  });

  return {
    createReviewAssessment: mutate,
    createReviewAssessmentAsync: mutateAsync,
    isCreatingAssessment: isLoading,
    error,
  };
};
