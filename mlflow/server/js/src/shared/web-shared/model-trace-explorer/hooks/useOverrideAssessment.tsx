import { omit } from 'lodash';

import { useIntl } from '@databricks/i18n';
import { getUser } from '@databricks/web-shared/global-settings';
import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import type { Assessment, Expectation, Feedback } from '../ModelTrace.types';
import { displayErrorNotification, FETCH_TRACE_INFO_QUERY_KEY } from '../ModelTraceExplorer.utils';
import type { CreateAssessmentPayload } from '../api';
import { createAssessment } from '../api';

export const useOverrideAssessment = ({
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

  const { mutate: overrideAssessmentMutation, isLoading } = useMutation({
    mutationFn: ({
      oldAssessment,
      value,
      rationale,
    }: {
      oldAssessment: Assessment;
      value: { feedback: Feedback } | { expectation: Expectation };
      rationale?: string;
    }) => {
      const newAssessment: Assessment = {
        ...oldAssessment,
        ...value,
        rationale,
        source: {
          source_id: getUser() ?? '',
          source_type: 'HUMAN',
        },
        overrides: oldAssessment.assessment_id,
      };
      const payload: CreateAssessmentPayload = {
        assessment: omit(newAssessment, 'assessment_id', 'create_time', 'last_update_time', 'overriddenAssessments'),
      };
      return createAssessment({ payload });
    },
    onError: (error) => {
      displayErrorNotification(
        intl.formatMessage(
          {
            defaultMessage: 'Failed to override assessment. Error: {error}',
            description: 'Error message when overriding an assessment fails',
          },
          {
            error: error instanceof Error ? error.message : String(error),
          },
        ),
      );
      onError?.(error);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [FETCH_TRACE_INFO_QUERY_KEY, traceId] });
      onSuccess?.();
    },
    onSettled: () => {
      onSettled?.();
    },
  });

  return {
    overrideAssessmentMutation,
    isLoading,
  };
};
