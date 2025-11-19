import { omit } from 'lodash';

import { useIntl } from '@databricks/i18n';
import { getUser } from '@databricks/web-shared/global-settings';
import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import { invalidateMlflowSearchTracesCache } from './invalidateMlflowSearchTracesCache';
import { shouldUseTracesV4API } from '../FeatureUtils';
import type { Assessment, Expectation, Feedback } from '../ModelTrace.types';
import {
  displayErrorNotification,
  doesTraceSupportV4API,
  FETCH_TRACE_INFO_QUERY_KEY,
  isV3ModelTraceInfo,
} from '../ModelTraceExplorer.utils';
import type { CreateAssessmentPayload, CreateAssessmentV3Response, CreateAssessmentV4Response } from '../api';
import { createAssessment, TracesServiceV4 } from '../api';
import { useModelTraceExplorerUpdateTraceContext } from '../contexts/UpdateTraceContext';

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

  const updateTraceVariables = useModelTraceExplorerUpdateTraceContext();

  const { mutate: overrideAssessmentMutation, isLoading } = useMutation({
    mutationFn: ({
      oldAssessment,
      value,
      rationale,
    }: {
      oldAssessment: Assessment;
      value: { feedback: Feedback } | { expectation: Expectation };
      rationale?: string;
    }): Promise<CreateAssessmentV4Response | CreateAssessmentV3Response> => {
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

      // TODO: Squash all this logic into a single util function (in both model-trace-explorer and genai-traces-table)
      if (
        shouldUseTracesV4API() &&
        updateTraceVariables.modelTraceInfo &&
        isV3ModelTraceInfo(updateTraceVariables.modelTraceInfo) &&
        doesTraceSupportV4API(updateTraceVariables.modelTraceInfo)
      ) {
        return TracesServiceV4.createAssessmentV4({
          payload,
          traceLocation: updateTraceVariables.modelTraceInfo.trace_location,
        });
      }

      return createAssessment({ payload });
    },
    onError: (error: any) => {
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
      invalidateMlflowSearchTracesCache({ queryClient });
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
