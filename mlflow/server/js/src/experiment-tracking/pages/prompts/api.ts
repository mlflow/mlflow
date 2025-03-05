import { fetchEndpoint } from '../../../common/utils/FetchUtils';
import { ModelVersionInfoEntity } from '../../types';
import { RegisteredPrompt, RegisteredPromptsListResponse } from './types';

const IS_PROMPT_TAG_NAME = 'mlflow.prompt.is_prompt';
const IS_PROMPT_TAG_VALUE = 'True';

const defaultErrorHandler = async ({ reject, response }: { reject: (cause: any) => void; response: Response }) => {
  // TODO: Add more detailed error handling
  const error = new Error('Request failed');
  if (response) {
    try {
      const messageFromReponse = (await response.json())?.message;
      error.message = messageFromReponse;
    } catch {
      // ignore
    }
  }

  reject(error);
};

export const RegisteredPromptsApi = {
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
    params.append('filter', `name='${promptName}'`);
    const relativeUrl = ['ajax-api/2.0/mlflow/model-versions/search', params.toString()].join('?');
    return fetchEndpoint({
      relativeUrl,
      error: defaultErrorHandler,
    }) as Promise<{
      model_versions?: ModelVersionInfoEntity[];
    }>;
  },
  listRegisteredPrompts: (searchFilter?: string, pageToken?: string) => {
    const params = new URLSearchParams();
    let filter = 'tags.`' + IS_PROMPT_TAG_NAME + "` = '" + IS_PROMPT_TAG_VALUE + "'";

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
  createRegisteredPromptVersion: (promptName: string) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/2.0/mlflow/model-versions/create',
      method: 'POST',
      body: JSON.stringify({ name: promptName, source: 'https://foobar/xyz' }),
      error: defaultErrorHandler,
    }) as Promise<{
      model_version?: ModelVersionInfoEntity;
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
  deleteRegisteredPromptTag: (promptName: string, key: string) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/2.0/mlflow/registered-models/delete-tag',
      method: 'DELETE',
      body: JSON.stringify({ key, name: promptName }),
      error: defaultErrorHandler,
    });
  },
  deleteRegisteredPrompt: (promptName: string) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/2.0/mlflow/registered-models/delete',
      method: 'DELETE',
      body: JSON.stringify({ name: promptName }),
      error: defaultErrorHandler,
    });
  },
};
