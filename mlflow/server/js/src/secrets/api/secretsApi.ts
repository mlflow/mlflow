import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import type {
  CreateSecretRequest,
  CreateSecretResponse,
  UpdateSecretRequest,
  DeleteSecretRequest,
  BindSecretRequest,
  UnbindSecretRequest,
  ListBindingsRequest,
  ListSecretsResponse,
  ListBindingsResponse,
  Secret,
} from '../types';

export const secretsApi = {
  listSecrets: async (): Promise<ListSecretsResponse> => {
    return (await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/secrets/list'), 'GET')) as ListSecretsResponse;
  },

  createSecret: async (request: CreateSecretRequest): Promise<CreateSecretResponse> => {
    return (await fetchAPI(
      getAjaxUrl('ajax-api/3.0/mlflow/secrets/create-and-bind'),
      'POST',
      request,
    )) as CreateSecretResponse;
  },

  updateSecret: async (request: UpdateSecretRequest): Promise<Secret> => {
    return (await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/secrets/update'), 'POST', request)) as Secret;
  },

  deleteSecret: async (request: DeleteSecretRequest): Promise<void> => {
    await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/secrets/delete'), 'DELETE', request);
  },

  listBindings: async (request: ListBindingsRequest): Promise<ListBindingsResponse> => {
    return (await fetchAPI(
      getAjaxUrl(`ajax-api/3.0/mlflow/secrets/list-bindings?secret_id=${request.secret_id}`),
      'GET',
    )) as ListBindingsResponse;
  },

  bindSecret: async (request: BindSecretRequest): Promise<void> => {
    await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/secrets/bind'), 'POST', request);
  },

  unbindSecret: async (request: UnbindSecretRequest): Promise<void> => {
    await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/secrets/unbind'), 'POST', request);
  },
};
