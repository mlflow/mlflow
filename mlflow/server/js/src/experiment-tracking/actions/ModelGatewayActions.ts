import { MlflowService } from 'experiment-tracking/sdk/MlflowService';
import { getUUID } from '../../common/utils/ActionUtils';
import type { AsyncAction } from '../../redux-types';
import {
  ModelGatewayQueryPayload,
  ModelGatewayRoute,
  ModelGatewayService,
  SearchModelGatewayRouteResponse,
  ModelGatewayResponseType,
} from '../sdk/ModelGatewayService';

export const SEARCH_MODEL_GATEWAY_ROUTES_API = 'SEARCH_MODEL_GATEWAY_ROUTES_API';
export interface SearchModelGatewayRoutesAction
  extends AsyncAction<SearchModelGatewayRouteResponse> {
  type: 'SEARCH_MODEL_GATEWAY_ROUTES_API';
}

// prettier-ignore
export const searchModelGatewayRoutesApi = (filter?: string): SearchModelGatewayRoutesAction => ({
  type: SEARCH_MODEL_GATEWAY_ROUTES_API,
  payload: MlflowService.gatewayProxyGet({
    gateway_path: 'api/2.0/gateway/routes/',
  }) as Promise<SearchModelGatewayRouteResponse>,
  meta: { id: getUUID() },
});

export const GET_MODEL_GATEWAY_ROUTE_API = 'GET_MODEL_GATEWAY_ROUTE_API';
export interface GetModelGatewayRouteAction extends AsyncAction<ModelGatewayRoute> {
  type: 'GET_MODEL_GATEWAY_ROUTE_API';
}

export const getModelGatewayRouteApi = (routeName: string): GetModelGatewayRouteAction => ({
  type: GET_MODEL_GATEWAY_ROUTE_API,
  payload: MlflowService.gatewayProxyGet({
    gateway_path: `api/2.0/gateway/routes/${routeName}`,
  }) as Promise<ModelGatewayRoute>,
  meta: { id: getUUID() },
});

export const QUERY_MODEL_GATEWAY_ROUTE_API = 'QUERY_MODEL_GATEWAY_ROUTE_API';

export const queryModelGatewayRouteApi = (
  route: ModelGatewayRoute,
  data: ModelGatewayQueryPayload,
) => {
  const { inputText } = data;
  const textPayload = ModelGatewayService.createEvaluationTextPayload(inputText, route);
  const processed_data = {
    ...textPayload,
    ...data.parameters,
  };

  return {
    type: QUERY_MODEL_GATEWAY_ROUTE_API,
    payload: MlflowService.gatewayProxyPost({
      gateway_path: `gateway/${route.name}/invocations`,
      json_data: processed_data,
    }) as Promise<ModelGatewayResponseType>,
    meta: { id: getUUID(), startTime: performance.now() },
  };
};
