import { MlflowService } from '@mlflow/mlflow/src/experiment-tracking/sdk/MlflowService';
import { getUUID } from '../../common/utils/ActionUtils';
import type { AsyncAction } from '../../redux-types';
import {
  ModelGatewayQueryPayload,
  ModelGatewayRoute,
  ModelGatewayRouteLegacy,
  ModelGatewayService,
  SearchMlflowDeploymentsModelRoutesResponse,
} from '../sdk/ModelGatewayService';

export const SEARCH_MLFLOW_DEPLOYMENTS_MODEL_ROUTES = 'SEARCH_MLFLOW_DEPLOYMENTS_MODEL_ROUTES';

export interface SearchMlflowDeploymentsModelRoutesAction
  extends AsyncAction<SearchMlflowDeploymentsModelRoutesResponse> {
  type: 'SEARCH_MLFLOW_DEPLOYMENTS_MODEL_ROUTES';
}

export const searchMlflowDeploymentsRoutesApi = (filter?: string): SearchMlflowDeploymentsModelRoutesAction => ({
  type: SEARCH_MLFLOW_DEPLOYMENTS_MODEL_ROUTES,
  payload: MlflowService.gatewayProxyGet({
    gateway_path: 'api/2.0/endpoints/',
  }) as Promise<SearchMlflowDeploymentsModelRoutesResponse>,
  meta: { id: getUUID() },
});
export const QUERY_MLFLOW_DEPLOYMENTS_ROUTE_API = 'QUERY_MLFLOW_DEPLOYMENTS_ROUTE_API';
export const queryMlflowDeploymentsRouteApi = (route: ModelGatewayRoute, data: ModelGatewayQueryPayload) => {
  return {
    type: QUERY_MLFLOW_DEPLOYMENTS_ROUTE_API,
    payload: ModelGatewayService.queryMLflowDeploymentEndpointRoute(route, data),
    meta: { id: getUUID(), startTime: performance.now() },
  };
};
