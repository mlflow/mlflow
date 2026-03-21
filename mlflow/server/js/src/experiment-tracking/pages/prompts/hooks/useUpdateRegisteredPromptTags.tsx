import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useEditKeyValueTagsModal } from '../../../../common/hooks/useEditKeyValueTagsModal';
import { RegisteredPromptsApi } from '../api';
import type { RegisteredPrompt } from '../types';
import { useCallback } from 'react';
import { diffCurrentAndNewTags, isUserFacingTag } from '../../../../common/utils/TagUtils';

type UpdateTagsPayload = {
  promptId: string;
  toAdd: { key: string; value: string }[];
  toDelete: { key: string }[];
};

export const useUpdateRegisteredPromptTags = ({ onSuccess }: { onSuccess?: () => void }) => {
  const updateMutation = useMutation<unknown, Error, UpdateTagsPayload>({
    mutationFn: async ({ toAdd, toDelete, promptId }) => {
      return Promise.all([
        ...toAdd.map(({ key, value }) => RegisteredPromptsApi.setRegisteredPromptTag(promptId, key, value)),
        ...toDelete.map(({ key }) => RegisteredPromptsApi.deleteRegisteredPromptTag(promptId, key)),
      ]);
    },
  });

  const { EditTagsModal, showEditTagsModal, isLoading } = useEditKeyValueTagsModal<
    Pick<RegisteredPrompt, 'name' | 'tags'>
  >({
    valueRequired: true,
    saveTagsHandler: (prompt, currentTags, newTags) => {
      const { addedOrModifiedTags, deletedTags } = diffCurrentAndNewTags(currentTags, newTags);

      return new Promise<void>((resolve, reject) => {
        if (!prompt.name) {
          return reject();
        }
        // Send all requests to the mutation
        updateMutation.mutate(
          {
            promptId: prompt.name,
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

  const showEditPromptTagsModal = useCallback(
    (prompt: RegisteredPrompt) =>
      showEditTagsModal({
        name: prompt.name,
        tags: prompt.tags.filter((tag) => isUserFacingTag(tag.key)),
      }),
    [showEditTagsModal],
  );

  return { EditTagsModal, showEditPromptTagsModal, isLoading };
};
