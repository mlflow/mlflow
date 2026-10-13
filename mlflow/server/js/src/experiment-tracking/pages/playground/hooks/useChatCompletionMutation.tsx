import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { PlaygroundApi } from '../api';
import type { ChatCompletionRequest, ChatCompletionResponse } from '../types';

export const useChatCompletionMutation = () => {
  return useMutation<ChatCompletionResponse, Error, ChatCompletionRequest>({
    mutationFn: (request) => PlaygroundApi.chatCompletion(request),
  });
};
