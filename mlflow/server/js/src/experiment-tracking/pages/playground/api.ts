import { fetchAPI, getAjaxUrl } from '../../../common/utils/FetchUtils';
import type { ChatCompletionRequest, ChatCompletionResponse } from './types';

export const PlaygroundApi = {
  chatCompletion: (request: ChatCompletionRequest) =>
    fetchAPI(getAjaxUrl('gateway/mlflow/v1/chat/completions'), {
      method: 'POST',
      body: request,
    }) as Promise<ChatCompletionResponse>,
};
