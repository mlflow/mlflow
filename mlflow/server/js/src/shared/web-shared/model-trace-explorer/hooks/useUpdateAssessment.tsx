import { useIntl } from '@databricks/i18n';
import { useMutation } from '@databricks/web-shared/query-client';

import type { Assessment } from '../ModelTrace.types';
import { displayErrorNotification } from '../ModelTraceExplorer.utils';
import type { UpdateAssessmentPayload } from '../api';
import { updateAssessment } from '../api';

// This API is used to update an assessment in place.
// To override an assessment (preserving the original)
// use `useOverrideAssessment` instead
export const useUpdateAssessment = ({
  assessment,
  refetchTraceInfo,
  onSuccess,
  onError,
  onSettled,
}: {
  assessment: Assessment;
  refetchTraceInfo: (() => Promise<any>) | null;
  onSuccess?: () => void;
  onError?: (error: any) => void;
  onSettled?: () => void;
}) => {
  const intl = useIntl();
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
      refetchTraceInfo?.();
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
