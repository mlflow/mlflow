import { isObject } from 'lodash';

/* eslint-disable import/no-duplicates */
import { useIntl } from '@databricks/i18n';
import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import { invalidateMlflowSearchTracesCache } from './invalidateMlflowSearchTracesCache';
import { useTraceCachedActions } from './useTraceCachedActions';
import { shouldUseTracesV4API } from '../FeatureUtils';
import { FETCH_TRACE_INFO_QUERY_KEY, displayErrorNotification, isV3ModelTraceInfo } from '../ModelTraceExplorer.utils';
import { doesTraceSupportV4API } from '../ModelTraceExplorer.utils';
import { createAssessment } from '../api';
import { TracesServiceV4 } from '../api';
import type { CreateAssessmentPayload, CreateAssessmentV3Response, CreateAssessmentV4Response } from '../api';
import { useModelTraceExplorerUpdateTraceContext } from '../contexts/UpdateTraceContext';

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

  const logCachedCreateAction = useTraceCachedActions((state) => state.logAddedAssessment);

  const updateTraceVariables = useModelTraceExplorerUpdateTraceContext();
  const { mutate: createAssessmentMutation, isLoading } = useMutation({
    mutationFn: (payload: CreateAssessmentPayload) => {
      // TODO: Squash all this logic into a single util function (in both model-trace-explorer and genai-traces-table)
      if (
        shouldUseTracesV4API() &&
        updateTraceVariables?.modelTraceInfo &&
        isV3ModelTraceInfo(updateTraceVariables.modelTraceInfo) &&
        doesTraceSupportV4API(updateTraceVariables?.modelTraceInfo)
      ) {
        return TracesServiceV4.createAssessmentV4({
          payload,
          traceLocation: updateTraceVariables.modelTraceInfo.trace_location,
        });
      }
      return createAssessment({ payload });
    },
    onSuccess: (createdAssessment: CreateAssessmentV4Response | CreateAssessmentV3Response) => {
      if (shouldUseTracesV4API() && isObject(createdAssessment)) {
        const assessment = 'assessment' in createdAssessment ? createdAssessment.assessment : createdAssessment;
        logCachedCreateAction(traceId, assessment);
      }
      queryClient.invalidateQueries({ queryKey: [FETCH_TRACE_INFO_QUERY_KEY, traceId] });
      invalidateMlflowSearchTracesCache({ queryClient });
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
