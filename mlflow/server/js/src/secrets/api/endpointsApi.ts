import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import type {
  CreateEndpointRequest,
  CreateEndpointResponse,
  UpdateEndpointRequest,
  UpdateEndpointResponse,
  ListEndpointsResponse,
  BindEndpointRequest,
  BindEndpointResponse,
} from '../types';

export const endpointsApi = {
  listEndpoints: async (): Promise<ListEndpointsResponse> => {
    return (await fetchAPI(
      getAjaxUrl('ajax-api/3.0/mlflow/secrets/list-routes'),
      'GET',
    )) as ListEndpointsResponse;
  },

  createEndpoint: async (request: CreateEndpointRequest): Promise<CreateEndpointResponse> => {
    return (await fetchAPI(
      getAjaxUrl('ajax-api/3.0/mlflow/secrets/endpoints/create'),
      'POST',
      request,
    )) as CreateEndpointResponse;
  },

  updateEndpoint: async (request: UpdateEndpointRequest): Promise<UpdateEndpointResponse> => {
    return (await fetchAPI(
      getAjaxUrl('ajax-api/3.0/mlflow/secrets/endpoints/update'),
      'PATCH',
      request,
    )) as UpdateEndpointResponse;
  },

  deleteEndpoint: async (endpointId: string): Promise<void> => {
    await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/secrets/endpoints/delete'), 'DELETE', {
      endpoint_id: endpointId,
    });
  },

  bindEndpoint: async (request: BindEndpointRequest): Promise<BindEndpointResponse> => {
    return (await fetchAPI(
      getAjaxUrl('ajax-api/3.0/mlflow/secrets/bind-route'),
      'POST',
      request,
    )) as BindEndpointResponse;
  },

  unbindEndpoint: async (bindingId: string): Promise<void> => {
    await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/secrets/bindings/delete'), 'DELETE', {
      binding_id: bindingId,
    });
  },

  listEndpointBindings: async (params: {
    endpoint_id?: string;
    secret_id?: string;
    resource_type?: string;
    resource_id?: string;
  }): Promise<import('../types').ListBindingsResponse> => {
    const queryParams = new URLSearchParams();
    if (params.endpoint_id) queryParams.append('endpoint_id', params.endpoint_id);
    if (params.secret_id) queryParams.append('secret_id', params.secret_id);
    if (params.resource_type) queryParams.append('resource_type', params.resource_type);
    if (params.resource_id) queryParams.append('resource_id', params.resource_id);

    return (await fetchAPI(
      getAjaxUrl(`ajax-api/3.0/mlflow/secrets/bindings/list?${queryParams.toString()}`),
      'GET',
    )) as import('../types').ListBindingsResponse;
  },
};
