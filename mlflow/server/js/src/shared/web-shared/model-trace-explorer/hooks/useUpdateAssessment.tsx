import { useIntl } from '@databricks/i18n';
import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import type { Assessment } from '../ModelTrace.types';
import { displayErrorNotification, FETCH_TRACE_INFO_QUERY_KEY } from '../ModelTraceExplorer.utils';
import type { UpdateAssessmentPayload } from '../api';
import { updateAssessment } from '../api';

// This API is used to update an assessment in place.
// To override an assessment (preserving the original)
// use `useOverrideAssessment` instead
export const useUpdateAssessment = ({
  assessment,
  onSuccess,
  onError,
  onSettled,
}: {
  assessment: Assessment;
  onSuccess?: () => void;
  onError?: (error: any) => void;
  onSettled?: () => void;
}) => {
  const intl = useIntl();
  const queryClient = useQueryClient();

  const { mutate: updateAssessmentMutation, isLoading } = useMutation({
    mutationFn: (payload: UpdateAssessmentPayload) =>
      updateAssessment({ traceId: assessment.trace_id, assessmentId: assessment.assessment_id, payload }),
    onError: (error) => {
      displayErrorNotification(
        intl.formatMessage(
          {
            defaultMessage: 'Failed to update assessment. Error: {error}',
            description: 'Error message when updating an assessment fails',
          },
          {
            error: error instanceof Error ? error.message : String(error),
          },
        ),
      );
      onError?.(error);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [FETCH_TRACE_INFO_QUERY_KEY, assessment.trace_id] });
      onSuccess?.();
    },
    onSettled: () => {
      onSettled?.();
    },
  });

  return {
    updateAssessmentMutation,
    isLoading,
  };
};
