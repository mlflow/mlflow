import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useEditKeyValueTagsModal } from '../../../../common/hooks/useEditKeyValueTagsModal';
import { RegisteredPromptsApi } from '../api';
import type { RegisteredPromptVersion } from '../types';
import { useCallback } from 'react';
import { diffCurrentAndNewTags, isUserFacingTag } from '../../../../common/utils/TagUtils';
import { FormattedMessage } from 'react-intl';

type UpdatePromptVersionMetadataPayload = {
  promptName: string;
  promptVersion: string;
  toAdd: { key: string; value: string }[];
  toDelete: { key: string }[];
};

export const useUpdatePromptVersionMetadataModal = ({ onSuccess }: { onSuccess?: () => void }) => {
  const updateMutation = useMutation<unknown, Error, UpdatePromptVersionMetadataPayload>({
    mutationFn: async ({ toAdd, toDelete, promptName, promptVersion }) => {
      return Promise.all([
        ...toAdd.map(({ key, value }) =>
          RegisteredPromptsApi.setRegisteredPromptVersionTag(promptName, promptVersion, key, value),
        ),
        ...toDelete.map(({ key }) =>
          RegisteredPromptsApi.deleteRegisteredPromptVersionTag(promptName, promptVersion, key),
        ),
      ]);
    },
  });

  const {
    EditTagsModal: EditPromptVersionMetadataModal,
    showEditTagsModal,
    isLoading,
  } = useEditKeyValueTagsModal<Pick<RegisteredPromptVersion, 'name' | 'version' | 'tags'>>({
    title: (
      <FormattedMessage
        defaultMessage="Add/Edit Prompt Version Metadata"
        description="Title for a modal that allows the user to add or edit metadata tags on prompt versions."
      />
    ),
    valueRequired: true,
    saveTagsHandler: (promptVersion, currentTags, newTags) => {
      const { addedOrModifiedTags, deletedTags } = diffCurrentAndNewTags(currentTags, newTags);

      return new Promise<void>((resolve, reject) => {
        if (!promptVersion.name) {
          return reject();
        }
        // Send all requests to the mutation
        updateMutation.mutate(
          {
            promptName: promptVersion.name,
            promptVersion: promptVersion.version,
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

  const showEditPromptVersionMetadataModal = useCallback(
    (promptVersion: RegisteredPromptVersion) =>
      showEditTagsModal({
        name: promptVersion.name,
        version: promptVersion.version,
        tags: promptVersion.tags?.filter((tag) => isUserFacingTag(tag.key)),
      }),
    [showEditTagsModal],
  );

  return { EditPromptVersionMetadataModal, showEditPromptVersionMetadataModal, isLoading };
};
