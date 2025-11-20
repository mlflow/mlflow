import type { Endpoint, EndpointModel, RouteResponse, CreateEndpointRequest_Legacy, CreateEndpointResponse_Legacy } from '../types';

/**
 * Transforms an Endpoint (multi-model) to a RouteResponse (single-model).
 * Since the API returns one route per model, we flatten endpoints to route responses.
 *
 * @param backendEndpoint - Endpoint with potentially multiple models
 * @param modelIndex - Index of the model to extract (defaults to first model)
 * @returns RouteResponse object compatible with legacy APIs
 */
export function backendEndpointToEndpoint(backendEndpoint: Endpoint, modelIndex: number = 0): RouteResponse {
  // Handle case where models array might be missing (e.g., in update responses)
  const models = backendEndpoint.models || [];
  const model: EndpointModel | undefined = models[modelIndex];

  if (!model) {
    // If no model is present, create a minimal endpoint object
    // This can happen when updating endpoint metadata only
    console.warn(
      `Endpoint ${backendEndpoint.endpoint_id} has no model at index ${modelIndex}. ` +
      `Using placeholder values.`
    );

    return {
      endpoint_id: backendEndpoint.endpoint_id,
      secret_id: '',
      secret_name: undefined,
      model_name: '',
      name: backendEndpoint.name,
      description: backendEndpoint.description,
      provider: undefined,
      created_at: backendEndpoint.created_at,
      last_updated_at: backendEndpoint.last_updated_at,
      created_by: backendEndpoint.created_by,
      last_updated_by: backendEndpoint.last_updated_by,
      tags: backendEndpoint.tags,
    };
  }

  return {
    endpoint_id: backendEndpoint.endpoint_id,
    secret_id: model.secret_id,
    secret_name: model.secret_name,
    model_name: model.model_name,
    name: backendEndpoint.name,
    description: backendEndpoint.description,
    provider: model.provider,
    created_at: backendEndpoint.created_at,
    last_updated_at: backendEndpoint.last_updated_at,
    created_by: backendEndpoint.created_by,
    last_updated_by: backendEndpoint.last_updated_by,
    tags: backendEndpoint.tags,
  };
}

/**
 * Transforms a list of Endpoints to RouteResponses.
 * For now, we only extract the first model from each endpoint.
 * Future: Could expand to show all models as separate route responses if needed.
 *
 * @param backendEndpoints - List of endpoints
 * @returns List of RouteResponse objects compatible with legacy APIs
 */
export function backendEndpointsToEndpoints(backendEndpoints: Endpoint[]): RouteResponse[] {
  return backendEndpoints.map((backendEndpoint) => backendEndpointToEndpoint(backendEndpoint, 0));
}

/**
 * Transforms a UI CreateRouteRequest to backend format.
 * UI sends single-model endpoint creation, backend expects endpoint with models array.
 *
 * @param request - UI endpoint creation request
 * @returns Object with separate API calls needed for backend
 */
export function routeRequestToBackendCalls(request: CreateEndpointRequest_Legacy): {
  needsSecretCreation: boolean;
  secretRequest?: {
    secret_name: string;
    secret_value: string;
    provider?: string;
    is_shared?: boolean;
    auth_config?: string;
  };
  endpointRequest: {
    name?: string;
    description?: string;
    tags?: Array<{ key: string; value: string }>;
    models: Array<{
      model_name: string;
      secret_id: string;
    }>;
    created_by?: string;
  };
  bindRequest?: {
    resource_type: string;
    resource_id: string;
    field_name: string;
  };
} {
  const needsSecretCreation = !request.secret_id && !!request.secret_name && !!request.secret_value;

  return {
    needsSecretCreation,
    secretRequest: needsSecretCreation
      ? {
          secret_name: request.secret_name!,
          secret_value: request.secret_value!,
          provider: request.provider,
          is_shared: request.is_shared,
          auth_config: request.auth_config,
        }
      : undefined,
    endpointRequest: {
      name: request.route_name,
      description: request.route_description,
      tags: request.route_tags ? JSON.parse(request.route_tags) : undefined,
      models: [
        {
          model_name: request.model_name,
          secret_id: request.secret_id || '', // Will be filled in after secret creation
        },
      ],
      created_by: request.created_by,
    },
    bindRequest:
      request.resource_type && request.resource_id && request.field_name
        ? {
            resource_type: request.resource_type,
            resource_id: request.resource_id,
            field_name: request.field_name,
          }
        : undefined,
  };
}

/**
 * Transforms a backend CreateEndpointResponse to UI CreateRouteResponse.
 *
 * @param backendEndpoint - Backend endpoint creation response
 * @param binding - Binding information (optional)
 * @returns Endpoint creation response compatible with UI
 */
export function endpointResponseToRouteResponse(
  backendEndpoint: Endpoint,
  binding?: { binding_id: string },
): CreateEndpointResponse_Legacy {
  const route = backendEndpointToEndpoint(backendEndpoint, 0);

  return {
    route,
    binding: binding
      ? {
          binding_id: binding.binding_id,
          secret_id: route.secret_id,
          resource_type: '', // Not included in bind response
          resource_id: '', // Not included in bind response
          field_name: '', // Not included in bind response
          created_at: Date.now() / 1000,
          last_updated_at: Date.now() / 1000,
          endpoint_id: route.endpoint_id,
          route_name: route.name,
          secret_name: route.secret_name,
          provider: route.provider,
        }
      : undefined,
  };
}
