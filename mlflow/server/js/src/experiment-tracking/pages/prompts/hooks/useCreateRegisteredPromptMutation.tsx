import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { RegisteredPromptsApi } from '../api';
import { REGISTERED_PROMPT_CONTENT_TAG_KEY } from '../utils';

type UpdateContentPayload = {
  promptName: string;
  createPromptEntity?: boolean;
  content: string;
  commitMessage?: string;
  tags: { key: string; value: string }[];
};

export const useCreateRegisteredPromptMutation = () => {
  const updateMutation = useMutation<{ version: string }, Error, UpdateContentPayload>({
    mutationFn: async ({ promptName, createPromptEntity, content, commitMessage, tags }) => {
      if (createPromptEntity) {
        await RegisteredPromptsApi.createRegisteredPrompt(promptName);
      }

      const version = await RegisteredPromptsApi.createRegisteredPromptVersion(
        promptName,
        [{ key: REGISTERED_PROMPT_CONTENT_TAG_KEY, value: content }, ...tags],
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
