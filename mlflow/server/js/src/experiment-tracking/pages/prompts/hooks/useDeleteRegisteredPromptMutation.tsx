import { useMutation } from '@tanstack/react-query';
import { RegisteredPromptsApi } from '../api';

type DeletePromptPayload = {
  promptName: string;
};

export const useDeleteRegisteredPromptMutation = () => {
  return useMutation<unknown, Error, DeletePromptPayload>({
    mutationFn: async ({ promptName }) => {
      await RegisteredPromptsApi.deleteRegisteredPrompt(promptName);
    },
  });
};
