import { combineReducers } from 'redux';
import { ModelGatewayRoute, SearchModelGatewayRouteResponse } from '../sdk/ModelGatewayService';
import { fulfilled, pending, rejected } from '../../common/utils/ActionUtils';

import { AsyncAction, AsyncFulfilledAction, Fulfilled } from '../../redux-types';
import {
  GET_MODEL_GATEWAY_ROUTE_API,
  GetModelGatewayRouteAction,
  SEARCH_MODEL_GATEWAY_ROUTES_API,
  SearchModelGatewayRoutesAction,
} from '../actions/ModelGatewayActions';

export interface ModelGatewayReduxState {
  modelGatewayRoutes: Record<string, ModelGatewayRoute>;
  modelGatewayRoutesLoading: boolean;
}

export const modelGatewayRoutesLoading = (
  state = false,
  action: AsyncAction<SearchModelGatewayRouteResponse>,
) => {
  switch (action.type) {
    case pending(SEARCH_MODEL_GATEWAY_ROUTES_API):
      return true;
    case fulfilled(SEARCH_MODEL_GATEWAY_ROUTES_API):
    case rejected(SEARCH_MODEL_GATEWAY_ROUTES_API):
      return false;
  }
  return state;
};

export const modelGatewayRoutes = (
  state: Record<string, ModelGatewayRoute> = {},
  {
    payload,
    type,
  }:
    | AsyncFulfilledAction<SearchModelGatewayRoutesAction>
    | AsyncFulfilledAction<GetModelGatewayRouteAction>,
): Record<string, ModelGatewayRoute> => {
  switch (type) {
    case fulfilled(SEARCH_MODEL_GATEWAY_ROUTES_API):
      if (!payload.routes) {
        return state;
      }
      return payload.routes.reduce(
        (newState, route) => ({ ...newState, [route.name]: route }),
        state,
      );
    case fulfilled(GET_MODEL_GATEWAY_ROUTE_API):
      return { ...state, [payload.name]: payload };
    default:
  }
  return state;
};

export const modelGatewayReducer = combineReducers({
  modelGatewayRoutesLoading,
  modelGatewayRoutes,
});
