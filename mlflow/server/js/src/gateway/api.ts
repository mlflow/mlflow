import { matchPredefinedError, UnknownError } from '@databricks/web-shared/errors';
import { fetchEndpoint } from '../common/utils/FetchUtils';
import type {
  ProvidersResponse,
  ModelsResponse,
  ProviderConfig,
  CreateSecretRequest,
  CreateSecretResponse,
  GetSecretResponse,
  UpdateSecretRequest,
  UpdateSecretResponse,
  ListSecretsResponse,
  CreateEndpointRequest,
  CreateEndpointResponse,
  GetEndpointResponse,
  UpdateEndpointRequest,
  UpdateEndpointResponse,
  ListEndpointsResponse,
  CreateModelDefinitionRequest,
  CreateModelDefinitionResponse,
  GetModelDefinitionResponse,
  ListModelDefinitionsResponse,
  UpdateModelDefinitionRequest,
  UpdateModelDefinitionResponse,
  AttachModelToEndpointRequest,
  AttachModelToEndpointResponse,
  DetachModelFromEndpointRequest,
  CreateEndpointBindingRequest,
  CreateEndpointBindingResponse,
  ListEndpointBindingsResponse,
} from './types';

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
      const messageFromResponse = (await response.json())?.message;
      if (messageFromResponse) {
        error.message = messageFromResponse;
      }
    } catch {
      // Keep original error message if extraction fails
    }
  }

  reject(error);
};

export const GatewayApi = {
  // Provider Metadata
  listProviders: () => {
    const relativeUrl = 'ajax-api/3.0/mlflow/endpoints/supported-providers';
    return fetchEndpoint({
      relativeUrl,
      error: defaultErrorHandler,
    }) as Promise<ProvidersResponse>;
  },

  listModels: (provider?: string) => {
    const params = new URLSearchParams();
    if (provider) {
      params.append('provider', provider);
    }
    const relativeUrl = ['ajax-api/3.0/mlflow/endpoints/supported-models', params.toString()].join('?');
    return fetchEndpoint({
      relativeUrl,
      error: defaultErrorHandler,
    }) as Promise<ModelsResponse>;
  },

  getProviderConfig: (provider: string) => {
    const params = new URLSearchParams();
    params.append('provider', provider);
    const relativeUrl = ['ajax-api/3.0/mlflow/endpoints/provider-config', params.toString()].join('?');
    return fetchEndpoint({
      relativeUrl,
      error: defaultErrorHandler,
    }) as Promise<ProviderConfig>;
  },

  // Secrets Management
  createSecret: (request: CreateSecretRequest) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/secrets/create',
      method: 'POST',
      body: JSON.stringify(request),
      error: defaultErrorHandler,
    }) as Promise<CreateSecretResponse>;
  },

  getSecret: (secretId: string) => {
    const params = new URLSearchParams();
    params.append('secret_id', secretId);
    const relativeUrl = ['ajax-api/3.0/mlflow/secrets/get', params.toString()].join('?');
    return fetchEndpoint({
      relativeUrl,
      error: defaultErrorHandler,
    }) as Promise<GetSecretResponse>;
  },

  updateSecret: (request: UpdateSecretRequest) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/secrets/update',
      method: 'POST',
      body: JSON.stringify(request),
      error: defaultErrorHandler,
    }) as Promise<UpdateSecretResponse>;
  },

  deleteSecret: (secretId: string) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/secrets/delete',
      method: 'DELETE',
      body: JSON.stringify({ secret_id: secretId }),
      error: defaultErrorHandler,
    });
  },

  listSecrets: (provider?: string) => {
    const params = new URLSearchParams();
    if (provider) {
      params.append('provider', provider);
    }
    const relativeUrl = ['ajax-api/3.0/mlflow/secrets/list', params.toString()].join('?');
    return fetchEndpoint({
      relativeUrl,
      error: defaultErrorHandler,
    }) as Promise<ListSecretsResponse>;
  },

  // Endpoints Management
  createEndpoint: (request: CreateEndpointRequest) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/endpoints/create',
      method: 'POST',
      body: JSON.stringify(request),
      error: defaultErrorHandler,
    }) as Promise<CreateEndpointResponse>;
  },

  getEndpoint: (endpointId: string) => {
    const params = new URLSearchParams();
    params.append('endpoint_id', endpointId);
    const relativeUrl = ['ajax-api/3.0/mlflow/endpoints/get', params.toString()].join('?');
    return fetchEndpoint({
      relativeUrl,
      error: defaultErrorHandler,
    }) as Promise<GetEndpointResponse>;
  },

  updateEndpoint: (request: UpdateEndpointRequest) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/endpoints/update',
      method: 'POST',
      body: JSON.stringify(request),
      error: defaultErrorHandler,
    }) as Promise<UpdateEndpointResponse>;
  },

  deleteEndpoint: (endpointId: string) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/endpoints/delete',
      method: 'DELETE',
      body: JSON.stringify({ endpoint_id: endpointId }),
      error: defaultErrorHandler,
    });
  },

  listEndpoints: (provider?: string) => {
    const params = new URLSearchParams();
    if (provider) {
      params.append('provider', provider);
    }
    const relativeUrl = ['ajax-api/3.0/mlflow/endpoints/list', params.toString()].join('?');
    return fetchEndpoint({
      relativeUrl,
      error: defaultErrorHandler,
    }) as Promise<ListEndpointsResponse>;
  },

  // Model Definitions Management
  createModelDefinition: (request: CreateModelDefinitionRequest) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/model-definitions/create',
      method: 'POST',
      body: JSON.stringify(request),
      error: defaultErrorHandler,
    }) as Promise<CreateModelDefinitionResponse>;
  },

  getModelDefinition: (modelDefinitionId: string) => {
    const params = new URLSearchParams();
    params.append('model_definition_id', modelDefinitionId);
    const relativeUrl = ['ajax-api/3.0/mlflow/model-definitions/get', params.toString()].join('?');
    return fetchEndpoint({
      relativeUrl,
      error: defaultErrorHandler,
    }) as Promise<GetModelDefinitionResponse>;
  },

  listModelDefinitions: () => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/model-definitions/list',
      error: defaultErrorHandler,
    }) as Promise<ListModelDefinitionsResponse>;
  },

  updateModelDefinition: (request: UpdateModelDefinitionRequest) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/model-definitions/update',
      method: 'POST',
      body: JSON.stringify(request),
      error: defaultErrorHandler,
    }) as Promise<UpdateModelDefinitionResponse>;
  },

  deleteModelDefinition: (modelDefinitionId: string) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/model-definitions/delete',
      method: 'DELETE',
      body: JSON.stringify({ model_definition_id: modelDefinitionId }),
      error: defaultErrorHandler,
    });
  },

  // Attach/Detach Models to Endpoints
  attachModelToEndpoint: (request: AttachModelToEndpointRequest) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/endpoints/models/attach',
      method: 'POST',
      body: JSON.stringify(request),
      error: defaultErrorHandler,
    }) as Promise<AttachModelToEndpointResponse>;
  },

  detachModelFromEndpoint: (request: DetachModelFromEndpointRequest) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/endpoints/models/detach',
      method: 'POST',
      body: JSON.stringify(request),
      error: defaultErrorHandler,
    });
  },

  // Endpoint Bindings Management
  createEndpointBinding: (request: CreateEndpointBindingRequest) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/endpoints/bindings/create',
      method: 'POST',
      body: JSON.stringify(request),
      error: defaultErrorHandler,
    }) as Promise<CreateEndpointBindingResponse>;
  },

  deleteEndpointBinding: (bindingId: string) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/endpoints/bindings/delete',
      method: 'DELETE',
      body: JSON.stringify({ binding_id: bindingId }),
      error: defaultErrorHandler,
    });
  },

  listEndpointBindings: (endpointId?: string, experimentId?: string) => {
    const params = new URLSearchParams();
    if (endpointId) {
      params.append('endpoint_id', endpointId);
    }
    if (experimentId) {
      params.append('experiment_id', experimentId);
    }
    const relativeUrl = ['ajax-api/3.0/mlflow/endpoints/bindings/list', params.toString()].join('?');
    return fetchEndpoint({
      relativeUrl,
      error: defaultErrorHandler,
    }) as Promise<ListEndpointBindingsResponse>;
  },
};
