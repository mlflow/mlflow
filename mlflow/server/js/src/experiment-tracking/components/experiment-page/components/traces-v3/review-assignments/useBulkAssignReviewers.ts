import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import { fetchAPI, getAjaxUrl } from '../../../../../../common/utils/FetchUtils';

interface BulkAssignReviewersParams {
  experimentId: string;
  traceIds: string[];
  reviewers: string[];
  assigner: string;
}

interface ReviewAssignmentProto {
  assignment_id: string;
  target_id: string;
  reviewer: string;
  state: string;
}

interface BulkCreateFailureProto {
  target_id: string;
  reviewer: string;
  error_message: string;
}

export interface BulkAssignReviewersResult {
  created: ReviewAssignmentProto[];
  existing: string[];
  failed: BulkCreateFailureProto[];
}

/**
 * Query key for per-target reviewer lookups. Shared so the reader (the
 * trace-detail "Reviewers" widget, a later stack) and this writer's
 * cache invalidation can't drift to different literals.
 */
export const REVIEW_ASSIGNMENTS_FOR_TARGET_QUERY_KEY = 'REVIEW_ASSIGNMENTS_FOR_TARGET';

/**
 * Bulk-assigns reviewers to traces by POSTing the cross product of
 * (traceIds x reviewers) to the review-assignments REST surface. The
 * response partitions every pair into created / existing / failed.
 */
export const useBulkAssignReviewers = () => {
  const queryClient = useQueryClient();
  return useMutation<BulkAssignReviewersResult, Error, BulkAssignReviewersParams>({
    mutationFn: async ({ experimentId, traceIds, reviewers, assigner }) => {
      const response = await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/review-assignments/bulk-create'), {
        method: 'POST',
        body: {
          experiment_id: experimentId,
          target_type: 'TRACE',
          target_ids: traceIds,
          reviewers,
          assigner,
        },
      });
      const result = response as Partial<BulkAssignReviewersResult>;
      return {
        created: result.created ?? [],
        existing: result.existing ?? [],
        failed: result.failed ?? [],
      };
    },
    onSuccess: () => {
      // The per-target "Reviewers" lookups are keyed by target id; drop
      // them so a freshly-assigned trace reflects the new reviewers.
      queryClient.invalidateQueries({ queryKey: [REVIEW_ASSIGNMENTS_FOR_TARGET_QUERY_KEY] });
    },
  });
};
