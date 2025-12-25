import { createMLflowRoutePath, generatePath } from '../common/utils/RoutingUtils';

export enum GatewayPageId {
  gatewayPage = 'mlflow.gateway',
  apiKeysPage = 'mlflow.gateway.api-keys',
  createEndpointPage = 'mlflow.gateway.create',
  endpointDetailsPage = 'mlflow.gateway.details',
  editEndpointPage = 'mlflow.gateway.edit',
}

export class GatewayRoutePaths {
  static get gatewayPage() {
    return createMLflowRoutePath('/gateway');
  }

  static get apiKeysPage() {
    return createMLflowRoutePath('/gateway/api-keys');
  }

  static get createEndpointPage() {
    return createMLflowRoutePath('/gateway/endpoints/create');
  }

  static get endpointDetailsPage() {
    return createMLflowRoutePath('/gateway/endpoints/:endpointId');
  }

  static get editEndpointPage() {
    return createMLflowRoutePath('/gateway/endpoints/:endpointId/edit');
  }
}

class GatewayRoutes {
  static get gatewayPageRoute() {
    return GatewayRoutePaths.gatewayPage;
  }

  static get apiKeysPageRoute() {
    return GatewayRoutePaths.apiKeysPage;
  }

  static get createEndpointPageRoute() {
    return GatewayRoutePaths.createEndpointPage;
  }

  static getEndpointDetailsRoute(endpointId: string) {
    return generatePath(GatewayRoutePaths.endpointDetailsPage, { endpointId });
  }

  static getEditEndpointRoute(endpointId: string) {
    return generatePath(GatewayRoutePaths.editEndpointPage, { endpointId });
  }
}

export default GatewayRoutes;
