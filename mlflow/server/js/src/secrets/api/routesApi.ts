import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import type {
  CreateRouteRequest,
  CreateRouteResponse,
  ListRoutesResponse,
  SecretBinding,
  UpdateRouteRequest,
  UpdateRouteResponse,
} from '../types';

export const routesApi = {
  listRoutes: async (): Promise<ListRoutesResponse> => {
    return (await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/secrets/list-routes'), 'GET')) as ListRoutesResponse;
  },

  createRoute: async (request: CreateRouteRequest): Promise<CreateRouteResponse> => {
    // Use create-and-bind endpoint for creating new secret + route
    // Use create-route-and-bind endpoint for binding route to existing secret
    const endpoint = request.secret_id
      ? 'ajax-api/3.0/mlflow/secrets/create-route-and-bind'
      : 'ajax-api/3.0/mlflow/secrets/create-and-bind';
    return (await fetchAPI(getAjaxUrl(endpoint), 'POST', request)) as CreateRouteResponse;
  },

  deleteRoute: async (routeId: string): Promise<void> => {
    await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/secrets/routes/delete'), 'DELETE', { route_id: routeId });
  },

  bindRoute: async (
    routeId: string,
    resourceType: string,
    resourceId: string,
    fieldName: string,
  ): Promise<SecretBinding> => {
    return (await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/secrets/routes/bind'), 'POST', {
      route_id: routeId,
      resource_type: resourceType,
      resource_id: resourceId,
      field_name: fieldName,
    })) as SecretBinding;
  },

  updateRoute: async (request: UpdateRouteRequest): Promise<UpdateRouteResponse> => {
    // Update route to point to a different secret (existing or new)
    // Preserves route_id and all bindings
    return (await fetchAPI(
      getAjaxUrl('ajax-api/3.0/mlflow/secrets/routes/update'),
      'PATCH',
      request,
    )) as UpdateRouteResponse;
  },
};
