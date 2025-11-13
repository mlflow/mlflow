import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import type { CreateRouteRequest, CreateRouteResponse, ListRoutesResponse } from '../types';

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
};
