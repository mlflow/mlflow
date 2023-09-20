import { fulfilled, pending } from '../../common/utils/ActionUtils';
import { AsyncAction, AsyncFulfilledAction } from '../../redux-types';
import {
  GET_MODEL_GATEWAY_ROUTE_API,
  GetModelGatewayRouteAction,
  SEARCH_MODEL_GATEWAY_ROUTES_API,
  SearchModelGatewayRoutesAction,
} from '../actions/ModelGatewayActions';
import { ModelGatewayRoute, ModelGatewayRouteType } from '../sdk/ModelGatewayService';
import { modelGatewayReducer } from './ModelGatewayReducer';

describe('modelGatewayReducer', () => {
  const emptyState: ReturnType<typeof modelGatewayReducer> = {
    modelGatewayRoutes: {},
    modelGatewayRoutesLoading: false,
  };

  const MOCK_DOLLY_MODEL_ROUTE = {
    model: { name: 'dolly-2', provider: 'mlflow' },
    name: 'test-dolly-gateway',
    route_type: ModelGatewayRouteType.LLM_V1_COMPLETIONS,
  };

  const MOCK_LLAMA_MODEL_ROUTE = {
    model: { name: 'llama-6', provider: 'meta' },
    name: 'test-llama-gateway',
    route_type: ModelGatewayRouteType.LLM_V1_COMPLETIONS,
  };

  const MOCK_GPT_MODEL_ROUTE = {
    model: { name: 'gpt-1', provider: 'gpt' },
    name: 'test-gpt-gateway',
    route_type: ModelGatewayRouteType.LLM_V1_COMPLETIONS,
  };

  const mockFulfilledSearchAction = (
    modelRoutes: ModelGatewayRoute[],
  ): AsyncFulfilledAction<SearchModelGatewayRoutesAction> => ({
    type: fulfilled(SEARCH_MODEL_GATEWAY_ROUTES_API),
    payload: { routes: modelRoutes },
  });

  const mockPendingSearchAction = (): AsyncAction => ({
    type: pending(SEARCH_MODEL_GATEWAY_ROUTES_API),
    payload: Promise.resolve(),
  });

  const mockFulfilledGetAction = (
    modelRoute: ModelGatewayRoute,
  ): AsyncFulfilledAction<GetModelGatewayRouteAction> => ({
    type: fulfilled(GET_MODEL_GATEWAY_ROUTE_API),
    payload: modelRoute,
  });

  it('model gateway entries are correctly populated by search action', () => {
    let state = emptyState;
    // Start searching for routes
    state = modelGatewayReducer(state, mockPendingSearchAction());
    expect(state.modelGatewayRoutesLoading).toEqual(true);

    // Search and retrieve 2 model routes
    state = modelGatewayReducer(
      state,
      mockFulfilledSearchAction([MOCK_DOLLY_MODEL_ROUTE, MOCK_GPT_MODEL_ROUTE]),
    );

    expect(state.modelGatewayRoutesLoading).toEqual(false);
    expect(state.modelGatewayRoutes['test-dolly-gateway'].model.name).toEqual('dolly-2');
    expect(state.modelGatewayRoutes['test-gpt-gateway'].model.name).toEqual('gpt-1');
    expect(Object.keys(state.modelGatewayRoutes)).toHaveLength(2);
  });

  it('"get" action properly append model routes to the store', () => {
    let state = emptyState;

    // Search and retrieve 2 model routes
    state = modelGatewayReducer(
      state,
      mockFulfilledSearchAction([MOCK_LLAMA_MODEL_ROUTE, MOCK_GPT_MODEL_ROUTE]),
    );
    expect(Object.keys(state.modelGatewayRoutes)).toHaveLength(2);

    // Fetch single model route already existing in the store
    state = modelGatewayReducer(state, mockFulfilledGetAction(MOCK_LLAMA_MODEL_ROUTE));
    expect(Object.keys(state.modelGatewayRoutes)).toHaveLength(2);

    // Fetch completely new model route
    state = modelGatewayReducer(state, mockFulfilledGetAction(MOCK_DOLLY_MODEL_ROUTE));
    expect(Object.keys(state.modelGatewayRoutes)).toHaveLength(3);
  });
});
