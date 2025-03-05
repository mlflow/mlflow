import { useMutation } from '@tanstack/react-query';
import { useEditKeyValueTagsModal } from '../../../../common/hooks/useEditKeyValueTagsModal';
import { RegisteredPromptsApi } from '../api';
import { RegisteredPrompt } from '../types';
import { useCallback } from 'react';
import { isUserFacingTag } from '../../../../common/utils/TagUtils';

type UpdateTagsPayload = {
  promptId: string;
  toAdd: { key: string; value: string }[];
  toDelete: { key: string }[];
};

export const useUpdateModelVersionTracesTagsModal = ({ onSuccess }: { onSuccess?: () => void }) => {
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
      // First, determine new tags to be added
      const addedOrModifiedTags = newTags.filter(
        ({ key: newTagKey, value: newTagValue }) =>
          !currentTags.some(
            ({ key: existingTagKey, value: existingTagValue }) =>
              existingTagKey === newTagKey && newTagValue === existingTagValue,
          ),
      );

      // Next, determine those to be deleted
      const deletedTags = currentTags.filter(
        ({ key: existingTagKey }) => !newTags.some(({ key: newTagKey }) => existingTagKey === newTagKey),
      );

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
