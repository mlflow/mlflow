import { fulfilled, pending } from '../../common/utils/ActionUtils';
import type { AsyncAction, AsyncFulfilledAction } from '../../redux-types';
import type { MlflowDeploymentsEndpoint } from '../sdk/ModelGatewayService';
import type { SearchMlflowDeploymentsModelRoutesAction } from '../actions/ModelGatewayActions';
import { ModelGatewayRouteTask } from '../sdk/MlflowEnums';
import { modelGatewayReducer } from './ModelGatewayReducer';

describe('modelGatewayReducer - MLflow deployments endpoints', () => {
  const emptyState: ReturnType<typeof modelGatewayReducer> = {
    modelGatewayRoutes: {},
    modelGatewayRoutesLoading: {
      deploymentRoutesLoading: false,
      endpointRoutesLoading: false,
      gatewayRoutesLoading: false,
      loading: false,
    },
  };

  const MOCK_MLFLOW_DEPLOYMENTS_RESPONSE: Partial<MlflowDeploymentsEndpoint>[] = [
    {
      endpoint_type: ModelGatewayRouteTask.LLM_V1_CHAT,
      name: 'test-mlflow-deployment-endpoint-chat',
      endpoint_url: 'http://deployment.server/endpoint-url',
      model: {
        name: 'mpt-3.5',
        provider: 'mosaic',
      },
    },
    {
      endpoint_type: ModelGatewayRouteTask.LLM_V1_EMBEDDINGS,
      name: 'test-mlflow-deployment-endpoint-embeddingss',
      endpoint_url: 'http://deployment.server/endpoint-url',
      model: {
        name: 'mpt-3.5',
        provider: 'mosaic',
      },
    },
  ];

  const mockFulfilledSearchDeploymentsAction = (
    endpoints: any,
  ): AsyncFulfilledAction<SearchMlflowDeploymentsModelRoutesAction> => ({
    type: fulfilled('SEARCH_MLFLOW_DEPLOYMENTS_MODEL_ROUTES'),
    payload: { endpoints },
  });

  const mockPendingSearchDeploymentsAction = (): AsyncAction => ({
    type: pending('SEARCH_MLFLOW_DEPLOYMENTS_MODEL_ROUTES'),
    payload: Promise.resolve(),
  });

  it('gateway routes are correctly populated by search action', () => {
    let state = emptyState;
    // Start searching for routes
    state = modelGatewayReducer(state, mockPendingSearchDeploymentsAction());
    expect(state.modelGatewayRoutesLoading.deploymentRoutesLoading).toEqual(true);
    expect(state.modelGatewayRoutesLoading.loading).toEqual(true);

    // Search and retrieve 2 model routes
    state = modelGatewayReducer(state, mockFulfilledSearchDeploymentsAction(MOCK_MLFLOW_DEPLOYMENTS_RESPONSE));

    expect(state.modelGatewayRoutesLoading.deploymentRoutesLoading).toEqual(false);
    expect(state.modelGatewayRoutesLoading.loading).toEqual(false);
    expect(state.modelGatewayRoutes['mlflow_deployment_endpoint:test-mlflow-deployment-endpoint-chat'].type).toEqual(
      'mlflow_deployment_endpoint',
    );
    expect(state.modelGatewayRoutes['mlflow_deployment_endpoint:test-mlflow-deployment-endpoint-chat'].name).toEqual(
      'test-mlflow-deployment-endpoint-chat',
    );
    expect(
      state.modelGatewayRoutes['mlflow_deployment_endpoint:test-mlflow-deployment-endpoint-chat'].mlflowDeployment,
    ).toEqual(MOCK_MLFLOW_DEPLOYMENTS_RESPONSE[0]);

    // We ignore embeddings endpoints for now
    expect(
      state.modelGatewayRoutes['mlflow_deployment_endpoint:test-mlflow-deployment-endpoint-embeddings'],
    ).toBeUndefined();
  });
});
