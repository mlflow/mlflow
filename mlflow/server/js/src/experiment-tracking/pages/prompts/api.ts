import { matchPredefinedError, UnknownError } from '@databricks/web-shared/errors';
import { fetchEndpoint } from '../../../common/utils/FetchUtils';
import type { RegisteredPrompt, RegisteredPromptsListResponse, RegisteredPromptVersion } from './types';
import { IS_PROMPT_TAG_NAME, IS_PROMPT_TAG_VALUE, REGISTERED_PROMPT_SOURCE_RUN_IDS } from './utils';

const defaultErrorHandler = async ({
  reject,
  response,
  err: originalError,
}: {
  reject: (cause: any) => void;
  response: Response;
  err: Error;
}) => {
  // Try to match the error to one of the predefined errors
  const predefinedError = matchPredefinedError(response);
  const error = predefinedError instanceof UnknownError ? originalError : predefinedError;
  if (response) {
    try {
      // Try to extract exact error message from the response
      const messageFromResponse = (await response.json())?.message;
      if (messageFromResponse) {
        error.message = messageFromResponse;
      }
    } catch {
      // If we fail to extract the message, we will keep the original error message
    }
  }

  reject(error);
};

export const RegisteredPromptsApi = {
  listRegisteredPrompts: (searchFilter?: string, pageToken?: string) => {
    const params = new URLSearchParams();
    let filter = `tags.\`${IS_PROMPT_TAG_NAME}\` = '${IS_PROMPT_TAG_VALUE}'`;

    if (searchFilter) {
      filter = `${filter} AND name ILIKE '%${searchFilter}%'`;
    }

    if (pageToken) {
      params.append('page_token', pageToken);
    }

    params.append('filter', filter);

    const relativeUrl = ['ajax-api/2.0/mlflow/registered-models/search', params.toString()].join('?');
    return fetchEndpoint({
      relativeUrl,
      error: defaultErrorHandler,
    }) as Promise<RegisteredPromptsListResponse>;
  },
  setRegisteredPromptTag: (promptName: string, key: string, value: string) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/2.0/mlflow/registered-models/set-tag',
      method: 'POST',
      body: JSON.stringify({ key, value, name: promptName }),
      error: defaultErrorHandler,
    });
  },
  deleteRegisteredPromptTag: (promptName: string, key: string) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/2.0/mlflow/registered-models/delete-tag',
      method: 'DELETE',
      body: JSON.stringify({ key, name: promptName }),
      error: defaultErrorHandler,
    });
  },
  createRegisteredPrompt: (promptName: string) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/2.0/mlflow/registered-models/create',
      method: 'POST',
      body: JSON.stringify({
        name: promptName,
        tags: [
          {
            key: IS_PROMPT_TAG_NAME,
            value: IS_PROMPT_TAG_VALUE,
          },
        ],
      }),
      error: defaultErrorHandler,
    }) as Promise<{
      registered_model?: RegisteredPrompt;
    }>;
  },
  createRegisteredPromptVersion: (
    promptName: string,
    tags: { key: string; value: string }[] = [],
    description?: string,
  ) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/2.0/mlflow/model-versions/create',
      method: 'POST',
      body: JSON.stringify({
        name: promptName,
        description,
        // Put a placeholder source here for now to satisfy the API validation
        // TODO: remove source after it's no longer needed
        source: 'dummy-source',
        tags: [
          {
            key: IS_PROMPT_TAG_NAME,
            value: IS_PROMPT_TAG_VALUE,
          },
          ...tags,
        ],
      }),
      error: defaultErrorHandler,
    }) as Promise<{
      model_version?: RegisteredPromptVersion;
    }>;
  },
  setRegisteredPromptVersionTag: (promptName: string, promptVersion: string, key: string, value: string) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/2.0/mlflow/model-versions/set-tag',
      method: 'POST',
      body: JSON.stringify({ key, value, name: promptName, version: promptVersion }),
      error: defaultErrorHandler,
    });
  },
  deleteRegisteredPromptVersionTag: (promptName: string, promptVersion: string, key: string) => {
    fetchEndpoint({
      relativeUrl: 'ajax-api/2.0/mlflow/model-versions/delete-tag',
      method: 'DELETE',
      body: JSON.stringify({ key, name: promptName, version: promptVersion }),
      error: defaultErrorHandler,
    });
  },
  getPromptDetails: (promptName: string) => {
    const params = new URLSearchParams();
    params.append('name', promptName);
    const relativeUrl = ['ajax-api/2.0/mlflow/registered-models/get', params.toString()].join('?');
    return fetchEndpoint({
      relativeUrl,
      error: defaultErrorHandler,
    }) as Promise<{
      registered_model: RegisteredPrompt;
    }>;
  },
  getPromptVersions: (promptName: string) => {
    const params = new URLSearchParams();
    params.append('filter', `name='${promptName}' AND tags.\`${IS_PROMPT_TAG_NAME}\` = '${IS_PROMPT_TAG_VALUE}'`);
    const relativeUrl = ['ajax-api/2.0/mlflow/model-versions/search', params.toString()].join('?');
    return fetchEndpoint({
      relativeUrl,
      error: defaultErrorHandler,
    }) as Promise<{
      model_versions?: RegisteredPromptVersion[];
    }>;
  },
  getPromptVersionsForRun: (runUuid: string) => {
    const params = new URLSearchParams();
    params.append(
      'filter',
      `tags.\`${IS_PROMPT_TAG_NAME}\` = '${IS_PROMPT_TAG_VALUE}' AND tags.\`${REGISTERED_PROMPT_SOURCE_RUN_IDS}\` ILIKE "%${runUuid}%"`,
    );
    const relativeUrl = ['ajax-api/2.0/mlflow/model-versions/search', params.toString()].join('?');
    return fetchEndpoint({
      relativeUrl,
      error: defaultErrorHandler,
    }) as Promise<{
      model_versions?: RegisteredPromptVersion[];
    }>;
  },
  deleteRegisteredPrompt: (promptName: string) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/2.0/mlflow/registered-models/delete',
      method: 'DELETE',
      body: JSON.stringify({ name: promptName }),
      error: defaultErrorHandler,
    });
  },
  deleteRegisteredPromptVersion: (promptName: string, version: string) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/2.0/mlflow/model-versions/delete',
      method: 'DELETE',
      body: JSON.stringify({ name: promptName, version }),
      error: defaultErrorHandler,
    });
  },
};
