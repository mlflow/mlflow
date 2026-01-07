import { createMLflowRoutePath, generatePath } from '../common/utils/RoutingUtils';

export enum GatewayPageId {
  gatewayPage = 'mlflow.gateway',
  apiKeysPage = 'mlflow.gateway.api-keys',
  createEndpointPage = 'mlflow.gateway.create',
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

  static get editEndpointPage() {
    return createMLflowRoutePath('/gateway/endpoints/:endpointId');
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

  static getEndpointRoute(endpointId: string) {
    return generatePath(GatewayRoutePaths.editEndpointPage, { endpointId });
  }

  static getEditEndpointRoute(endpointId: string) {
    return generatePath(GatewayRoutePaths.editEndpointPage, { endpointId });
  }

  // Deprecated: Use getEndpointRoute instead
  static getEndpointDetailsRoute(endpointId: string) {
    return this.getEndpointRoute(endpointId);
  }
}

export default GatewayRoutes;
