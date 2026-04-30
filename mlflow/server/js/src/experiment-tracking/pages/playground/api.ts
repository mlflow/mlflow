import { matchPredefinedError, UnknownError } from '@databricks/web-shared/errors';
import { fetchEndpoint } from '../../../common/utils/FetchUtils';
import type { ChatCompletionRequest, ChatCompletionResponse } from './types';

const defaultErrorHandler = async ({
  reject,
  response,
  err: originalError,
}: {
  reject: (cause: any) => void;
  response: Response;
  err: Error;
}) => {
  const predefinedError = matchPredefinedError(response);
  const error = predefinedError instanceof UnknownError ? originalError : predefinedError;
  if (response) {
    try {
      const body = await response.json();
      const messageFromResponse = body?.message ?? body?.detail;
      if (typeof messageFromResponse === 'string') {
        error.message = messageFromResponse;
      }
    } catch {
      // Keep original error message if extraction fails
    }
  }

  reject(error);
};

export const PlaygroundApi = {
  chatCompletion: (request: ChatCompletionRequest) => {
    return fetchEndpoint({
      relativeUrl: 'gateway/mlflow/v1/chat/completions',
      method: 'POST',
      body: JSON.stringify(request),
      error: defaultErrorHandler,
    }) as Promise<ChatCompletionResponse>;
  },
};
