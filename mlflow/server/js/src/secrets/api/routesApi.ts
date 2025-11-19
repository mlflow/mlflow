import type {
  CreateEndpointRequest_Legacy,
  CreateEndpointResponse_Legacy,
  ListEndpointsResponse_Legacy,
  SecretBinding,
  UpdateEndpointRequest_Legacy,
  UpdateEndpointResponse_Legacy,
} from '../types';
import { endpointsApi } from './endpointsApi';
import { secretsApi } from './secretsApi';
import {
  backendEndpointsToEndpoints,
  routeRequestToBackendCalls,
  endpointResponseToRouteResponse,
  backendEndpointToEndpoint,
} from './endpointAdapters';

export const endpointsApi_Legacy = {
  listEndpoints: async (): Promise<ListEndpointsResponse_Legacy> => {
    const response = await endpointsApi.listEndpoints();
    return {
      routes: backendEndpointsToEndpoints(response.routes || []),
    };
  },

  createEndpoint: async (request: CreateEndpointRequest_Legacy): Promise<CreateEndpointResponse_Legacy> => {
    const { needsSecretCreation, secretRequest, endpointRequest, bindRequest } = routeRequestToBackendCalls(request);

    let secretId = request.secret_id || '';

    if (needsSecretCreation && secretRequest) {
      const secretResponse = await secretsApi.createSecret({
        secret_name: secretRequest.secret_name,
        secret_value: secretRequest.secret_value,
        provider: secretRequest.provider,
        is_shared: secretRequest.is_shared,
        created_by: request.created_by,
        auth_config: secretRequest.auth_config,
      });
      secretId = secretResponse.secret.secret_id;
    }

    endpointRequest.models[0].secret_id = secretId;

    const endpointResponse = await endpointsApi.createEndpoint(endpointRequest);

    let binding;
    if (bindRequest) {
      const bindResponse = await endpointsApi.bindEndpoint({
        endpoint_id: endpointResponse.endpoint.endpoint_id,
        resource_type: bindRequest.resource_type,
        resource_id: bindRequest.resource_id,
        field_name: bindRequest.field_name,
      });
      binding = bindResponse;
    }

    return endpointResponseToRouteResponse(endpointResponse.endpoint, binding);
  },

  deleteEndpoint: async (endpointId: string): Promise<void> => {
    await endpointsApi.deleteEndpoint(endpointId);
  },

  bindEndpoint: async (
    endpointId: string,
    resourceType: string,
    resourceId: string,
    fieldName: string,
  ): Promise<SecretBinding> => {
    const bindResponse = await endpointsApi.bindEndpoint({
      endpoint_id: endpointId,
      resource_type: resourceType,
      resource_id: resourceId,
      field_name: fieldName,
    });

    return {
      binding_id: bindResponse.binding_id,
      secret_id: '',
      resource_type: resourceType,
      resource_id: resourceId,
      field_name: fieldName,
      created_at: Date.now() / 1000,
      last_updated_at: Date.now() / 1000,
    };
  },

  updateEndpoint: async (request: UpdateEndpointRequest_Legacy): Promise<UpdateEndpointResponse_Legacy> => {
    let secretId = request.secret_id;

    if (!secretId && request.secret_name && request.secret_value) {
      const secretResponse = await secretsApi.createSecret({
        secret_name: request.secret_name,
        secret_value: request.secret_value,
        provider: request.provider,
        is_shared: request.is_shared,
        auth_config: request.auth_config,
      });
      secretId = secretResponse.secret.secret_id;
    }

    const updateResponse = await endpointsApi.updateEndpoint({
      endpoint_id: request.endpoint_id,
      name: request.route_description,
      description: request.route_description,
      tags: request.route_tags ? JSON.parse(request.route_tags) : undefined,
    });

    const route = backendEndpointToEndpoint(updateResponse.endpoint, 0);

    return {
      route,
      secret: {
        secret_id: secretId || route.secret_id,
        secret_name: request.secret_name || route.secret_name || '',
        masked_value: '***',
        is_shared: request.is_shared || false,
        created_at: Date.now() / 1000,
        last_updated_at: Date.now() / 1000,
      },
    };
  },
};
