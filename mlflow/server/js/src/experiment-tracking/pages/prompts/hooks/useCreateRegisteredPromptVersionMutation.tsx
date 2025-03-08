import { useMutation } from '@tanstack/react-query';
import { RegisteredPromptsApi } from '../api';
import { REGISTERED_PROMPT_CONTENT_TAG_KEY } from '../utils';

type UpdateContentPayload = {
  promptName: string;
  createPromptEntity?: boolean;
  content: string;
};

export const useCreateRegisteredPromptVersionMutation = () => {
  const updateMutation = useMutation<{ version: string }, Error, UpdateContentPayload>({
    mutationFn: async ({ promptName, createPromptEntity, content }) => {
      if (createPromptEntity) {
        await RegisteredPromptsApi.createRegisteredPrompt(promptName);
      }
      const version = await RegisteredPromptsApi.createRegisteredPromptVersion(promptName);
      const newVersionNumber = version?.model_version?.version;
      if (!newVersionNumber) {
        throw new Error('Failed to create a new prompt version');
      }
      await RegisteredPromptsApi.setRegisteredPromptVersionTag(
        promptName,
        newVersionNumber,
        REGISTERED_PROMPT_CONTENT_TAG_KEY,
        content,
      );
      return { version: newVersionNumber };
    },
  });

  return updateMutation;
};
