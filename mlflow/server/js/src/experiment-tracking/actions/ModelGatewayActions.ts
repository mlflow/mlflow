import { MlflowService } from '@mlflow/mlflow/src/experiment-tracking/sdk/MlflowService';
import { getUUID } from '../../common/utils/ActionUtils';
import type { AsyncAction } from '../../redux-types';
import type {
  ModelGatewayQueryPayload,
  ModelGatewayRouteLegacy,
  SearchMlflowDeploymentsModelRoutesResponse,
} from '../sdk/ModelGatewayService';
import { ModelGatewayRoute, ModelGatewayService } from '../sdk/ModelGatewayService';

const SEARCH_MLFLOW_DEPLOYMENTS_MODEL_ROUTES = 'SEARCH_MLFLOW_DEPLOYMENTS_MODEL_ROUTES';

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
