import { useIntl } from '@databricks/i18n';
import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import { FETCH_TRACE_INFO_QUERY_KEY, displayErrorNotification } from '../ModelTraceExplorer.utils';
import { createAssessment } from '../api';
import type { CreateAssessmentPayload } from '../api';

export const useCreateAssessment = ({
  traceId,
  onSuccess,
  onError,
  onSettled,
}: {
  traceId: string;
  onSuccess?: () => void;
  onError?: (error: any) => void;
  onSettled?: () => void;
}) => {
  const intl = useIntl();
  const queryClient = useQueryClient();
  const { mutate: createAssessmentMutation, isLoading } = useMutation({
    mutationFn: (payload: CreateAssessmentPayload) => createAssessment({ payload }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [FETCH_TRACE_INFO_QUERY_KEY, traceId] });
      onSuccess?.();
    },
    onError: (error) => {
      displayErrorNotification(
        intl.formatMessage(
          {
            defaultMessage: 'Failed to create assessment. Error: {error}',
            description: 'Error message when creating an assessment fails',
          },
          {
            error: error instanceof Error ? error.message : String(error),
          },
        ),
      );
      onError?.(error);
    },
    onSettled: () => {
      onSettled?.();
    },
  });

  return {
    createAssessmentMutation,
    isLoading,
  };
};
