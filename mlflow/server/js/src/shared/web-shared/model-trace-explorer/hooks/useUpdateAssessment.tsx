import { isObject } from 'lodash';

import { useIntl } from '@databricks/i18n';
import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import { invalidateMlflowSearchTracesCache } from './invalidateMlflowSearchTracesCache';
import { useTraceCachedActions } from './useTraceCachedActions';
import { shouldUseTracesV4API } from '../FeatureUtils';
import type { Assessment } from '../ModelTrace.types';
import {
  displayErrorNotification,
  FETCH_TRACE_INFO_QUERY_KEY,
  isV3ModelTraceInfo,
  doesTraceSupportV4API,
} from '../ModelTraceExplorer.utils';
import type { UpdateAssessmentPayload, UpdateAssessmentV3Response, UpdateAssessmentV4Response } from '../api';
import { updateAssessment, TracesServiceV4 } from '../api';
import { useModelTraceExplorerUpdateTraceContext } from '../contexts/UpdateTraceContext';

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

  const logCachedUpdateAction = useTraceCachedActions((state) => state.logAddedAssessment);

  const updateTraceVariables = useModelTraceExplorerUpdateTraceContext();

  const { mutate: updateAssessmentMutation, isLoading } = useMutation({
    mutationFn: (payload: UpdateAssessmentPayload) => {
      // TODO: Squash all this logic into a single util function (in both model-trace-explorer and genai-traces-table)
      if (
        shouldUseTracesV4API() &&
        updateTraceVariables?.modelTraceInfo &&
        isV3ModelTraceInfo(updateTraceVariables.modelTraceInfo) &&
        doesTraceSupportV4API(updateTraceVariables?.modelTraceInfo)
      ) {
        // V4 API requires assessment_name to be set on update
        payload.assessment.assessment_name = assessment.assessment_name;
        return TracesServiceV4.updateAssessmentV4({
          traceId: assessment.trace_id,
          assessmentId: assessment.assessment_id,
          payload,
          traceLocation: updateTraceVariables.modelTraceInfo.trace_location,
        });
      }
      return updateAssessment({ traceId: assessment.trace_id, assessmentId: assessment.assessment_id, payload });
    },
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
    onSuccess: (updatedAssessment: UpdateAssessmentV3Response | UpdateAssessmentV4Response) => {
      if (shouldUseTracesV4API() && isObject(updatedAssessment)) {
        const assessment = 'assessment' in updatedAssessment ? updatedAssessment.assessment : updatedAssessment;
        logCachedUpdateAction(assessment.trace_id, assessment);
      }
      queryClient.invalidateQueries({ queryKey: [FETCH_TRACE_INFO_QUERY_KEY, assessment.trace_id] });
      invalidateMlflowSearchTracesCache({ queryClient });
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
