import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { RegisteredPromptsApi } from '../api';
import type { PROMPT_TYPE_CHAT, PROMPT_TYPE_TEXT } from '../utils';
import { PROMPT_TYPE_TAG_KEY, REGISTERED_PROMPT_CONTENT_TAG_KEY } from '../utils';

type UpdateContentPayload = {
  promptName: string;
  promptType: typeof PROMPT_TYPE_CHAT | typeof PROMPT_TYPE_TEXT;
  createPromptEntity?: boolean;
  content: string;
  commitMessage?: string;
  tags: { key: string; value: string }[];
  promptTags?: { key: string; value: string }[];
};

export const useCreateRegisteredPromptMutation = () => {
  const updateMutation = useMutation<{ version: string }, Error, UpdateContentPayload>({
    mutationFn: async ({
      promptName,
      promptType,
      createPromptEntity,
      content,
      commitMessage,
      tags,
      promptTags = [],
    }) => {
      if (createPromptEntity) {
        await RegisteredPromptsApi.createRegisteredPrompt(promptName, promptTags);
      }

      const version = await RegisteredPromptsApi.createRegisteredPromptVersion(
        promptName,
        [
          { key: REGISTERED_PROMPT_CONTENT_TAG_KEY, value: content },
          { key: PROMPT_TYPE_TAG_KEY, value: promptType },
          ...tags,
        ],
        commitMessage,
      );

      const newVersionNumber = version?.model_version?.version;
      if (!newVersionNumber) {
        throw new Error('Failed to create a new prompt version');
      }
      return { version: newVersionNumber };
    },
  });

  return updateMutation;
};
