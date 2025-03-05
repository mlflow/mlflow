import { useMutation } from '@tanstack/react-query';
import { RegisteredPromptsApi } from '../api';
import { REGISTERED_PROMPT_CONTENT_TAG_KEY } from '../utils';

type UpdateContentPayload = {
  promptName: string;
  promptVersion: string;
  content: string;
};

export const useUpdateRegisteredPromptContentMutation = () => {
  const updateMutation = useMutation<unknown, Error, UpdateContentPayload>({
    mutationFn: async ({ promptName, promptVersion, content }) => {
      return RegisteredPromptsApi.setRegisteredPromptVersionTag(
        promptName,
        promptVersion,
        REGISTERED_PROMPT_CONTENT_TAG_KEY,
        content,
      );
    },
  });

  return updateMutation;
};
