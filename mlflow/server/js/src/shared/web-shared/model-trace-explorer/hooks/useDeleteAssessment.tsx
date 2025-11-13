import { useIntl } from '@databricks/i18n';
import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import { invalidateMlflowSearchTracesCache } from './invalidateMlflowSearchTracesCache';
import { useTraceCachedActions } from './useTraceCachedActions';
import { shouldUseTracesV4API } from '../FeatureUtils';
import type { Assessment } from '../ModelTrace.types';
import {
  displayErrorNotification,
  doesTraceSupportV4API,
  FETCH_TRACE_INFO_QUERY_KEY,
  isV3ModelTraceInfo,
} from '../ModelTraceExplorer.utils';
import { deleteAssessment, TracesServiceV4 } from '../api';
import { useModelTraceExplorerUpdateTraceContext } from '../contexts/UpdateTraceContext';

export const useDeleteAssessment = ({
  assessment,
  onSuccess,
  onError,
  onSettled,
  skip,
}: {
  assessment?: Assessment;
  onSuccess?: () => void;
  onError?: (error: any) => void;
  onSettled?: () => void;
  skip?: boolean;
}) => {
  const intl = useIntl();
  const queryClient = useQueryClient();
  const traceId = assessment?.trace_id;
  const assessmentId = assessment?.assessment_id;

  const logCachedDeleteAction = useTraceCachedActions((state) => state.logRemovedAssessment);

  const updateTraceVariables = useModelTraceExplorerUpdateTraceContext();
  const isSkipped = skip || !traceId || !assessmentId;

  const { mutate: deleteAssessmentMutation, isLoading } = useMutation({
    mutationFn: () => {
      if (isSkipped) {
        return Promise.reject(new Error('Mutation is skipped'));
      }

      if (
        shouldUseTracesV4API() &&
        updateTraceVariables.modelTraceInfo &&
        isV3ModelTraceInfo(updateTraceVariables.modelTraceInfo) &&
        doesTraceSupportV4API(updateTraceVariables.modelTraceInfo)
      ) {
        return TracesServiceV4.deleteAssessmentV4({
          traceId,
          assessmentId,
          traceLocation: updateTraceVariables.modelTraceInfo.trace_location,
        });
      }
      return deleteAssessment({ traceId, assessmentId });
    },
    onSuccess: () => {
      if (shouldUseTracesV4API() && !isSkipped) {
        logCachedDeleteAction(traceId, assessment);
      }
      queryClient.invalidateQueries({ queryKey: [FETCH_TRACE_INFO_QUERY_KEY, traceId] });
      invalidateMlflowSearchTracesCache({ queryClient });
      onSuccess?.();
    },
    onError: (error) => {
      if (error instanceof Error && isSkipped) {
        return;
      }

      displayErrorNotification(
        intl.formatMessage(
          {
            defaultMessage: 'Failed to delete assessment. Error: {error}',
            description: 'Error message when deleting an assessment fails.',
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
    deleteAssessmentMutation,
    isLoading,
  };
};
