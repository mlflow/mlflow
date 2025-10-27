import { type ModelTraceInfo } from '@databricks/web-shared/model-trace-explorer';
import { useEditKeyValueTagsModal } from '../../../../common/hooks/useEditKeyValueTagsModal';
import { MlflowService } from '../../../sdk/MlflowService';
import type { KeyValueEntity } from '../../../../common/types';
import { useCallback } from 'react';
import { MLFLOW_INTERNAL_PREFIX } from '../../../../common/utils/TagUtils';

type EditedModelTrace = {
  traceRequestId: string;
  tags: KeyValueEntity[];
};

export const useEditExperimentTraceTags = ({
  onSuccess,
  existingTagKeys = [],
  useV3Apis,
}: {
  onSuccess?: () => void;
  existingTagKeys?: string[];
  useV3Apis?: boolean;
}) => {
  const { showEditTagsModal, EditTagsModal } = useEditKeyValueTagsModal<EditedModelTrace>({
    saveTagsHandler: async (editedEntity, existingTags, newTags) => {
      if (!editedEntity.traceRequestId) {
        return;
      }
      const requestId = editedEntity.traceRequestId;
      // First, determine new tags to be added
      const addedOrModifiedTags = newTags.filter(
        ({ key: newTagKey, value: newTagValue }) =>
          !existingTags.some(
            ({ key: existingTagKey, value: existingTagValue }) =>
              existingTagKey === newTagKey && newTagValue === existingTagValue,
          ),
      );

      // Next, determine those to be deleted
      const deletedTags = existingTags.filter(
        ({ key: existingTagKey }) => !newTags.some(({ key: newTagKey }) => existingTagKey === newTagKey),
      );

      // Fire all requests at once
      const updateRequests = Promise.all([
        ...addedOrModifiedTags.map(({ key, value }) =>
          useV3Apis
            ? MlflowService.setExperimentTraceTagV3(requestId, key, value)
            : MlflowService.setExperimentTraceTag(requestId, key, value),
        ),
        ...deletedTags.map(({ key }) =>
          useV3Apis
            ? MlflowService.deleteExperimentTraceTagV3(requestId, key)
            : MlflowService.deleteExperimentTraceTag(requestId, key),
        ),
      ]);

      return updateRequests;
    },
    valueRequired: true,
    allAvailableTags: existingTagKeys.filter((tagKey) => tagKey && !tagKey.startsWith(MLFLOW_INTERNAL_PREFIX)),
    onSuccess: onSuccess,
  });

  const showEditTagsModalForTrace = useCallback(
    (trace: ModelTraceInfo) => {
      if (!trace.request_id) {
        return;
      }
      const visibleTags = trace.tags?.filter(({ key }) => key && !key.startsWith(MLFLOW_INTERNAL_PREFIX)) || [];
      showEditTagsModal({
        traceRequestId: trace.request_id,
        tags: visibleTags || [],
      });
    },
    [showEditTagsModal],
  );

  return {
    showEditTagsModalForTrace,
    EditTagsModal,
  };
};
