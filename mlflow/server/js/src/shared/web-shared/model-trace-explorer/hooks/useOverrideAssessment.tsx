import { omit } from 'lodash';

import { useIntl } from '@databricks/i18n';
import { useMutation } from '@databricks/web-shared/query-client';

import type { Assessment, Expectation, Feedback } from '../ModelTrace.types';
import { displayErrorNotification, getCurrentUser } from '../ModelTraceExplorer.utils';
import type { CreateAssessmentPayload } from '../api';
import { createAssessment } from '../api';

export const useOverrideAssessment = ({
  refetchTraceInfo,
  onSuccess,
  onError,
  onSettled,
}: {
  refetchTraceInfo: (() => Promise<any>) | null;
  onSuccess?: () => void;
  onError?: (error: any) => void;
  onSettled?: () => void;
}) => {
  const intl = useIntl();

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
          source_id: getCurrentUser(),
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
      refetchTraceInfo?.();
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
