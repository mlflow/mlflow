import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useEditKeyValueTagsModal } from '../../../../common/hooks/useEditKeyValueTagsModal';
import { useCallback } from 'react';
import { diffCurrentAndNewTags, isUserFacingTag } from '../../../../common/utils/TagUtils';
import { MlflowService } from '../../../sdk/MlflowService';
import type { ExperimentEntity } from '../../../types';

type UpdateTagsPayload = {
  experimentId: string;
  toAdd: { key: string; value: string }[];
  toDelete: { key: string }[];
};

export const useUpdateExperimentTags = ({ onSuccess }: { onSuccess?: () => void }) => {
  const updateMutation = useMutation<unknown, Error, UpdateTagsPayload>({
    mutationFn: async ({ toAdd, toDelete, experimentId }) => {
      return Promise.all([
        ...toAdd.map(({ key, value }) => MlflowService.setExperimentTag({ experiment_id: experimentId, key, value })),
        ...toDelete.map(({ key }) => MlflowService.deleteExperimentTag({ experiment_id: experimentId, key })),
      ]);
    },
  });

  const { EditTagsModal, showEditTagsModal, isLoading } = useEditKeyValueTagsModal<
    Pick<ExperimentEntity, 'experimentId' | 'name' | 'tags'>
  >({
    valueRequired: true,
    saveTagsHandler: (experiment, currentTags, newTags) => {
      const { addedOrModifiedTags, deletedTags } = diffCurrentAndNewTags(currentTags, newTags);

      return new Promise<void>((resolve, reject) => {
        if (!experiment) {
          return reject();
        }
        // Send all requests to the mutation
        updateMutation.mutate(
          {
            experimentId: experiment.experimentId,
            toAdd: addedOrModifiedTags,
            toDelete: deletedTags,
          },
          {
            onSuccess: () => {
              resolve();
              onSuccess?.();
            },
            onError: reject,
          },
        );
      });
    },
  });

  const showEditExperimentTagsModal = useCallback(
    (experiment: ExperimentEntity) =>
      showEditTagsModal({
        experimentId: experiment.experimentId,
        name: experiment.name,
        tags: experiment.tags.filter((tag) => isUserFacingTag(tag.key)),
      }),
    [showEditTagsModal],
  );

  return { EditTagsModal, showEditExperimentTagsModal, isLoading };
};
