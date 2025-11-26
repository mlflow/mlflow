import { createMLflowRoutePath, generatePath } from '../common/utils/RoutingUtils';

export enum GatewayPageId {
  gatewayPage = 'mlflow.gateway',
  createEndpointPage = 'mlflow.gateway.create',
  endpointDetailsPage = 'mlflow.gateway.details',
  editEndpointPage = 'mlflow.gateway.edit',
}

export class GatewayRoutePaths {
  static get gatewayPage() {
    return createMLflowRoutePath('/gateway');
  }

  static get createEndpointPage() {
    return createMLflowRoutePath('/gateway/create');
  }

  static get endpointDetailsPage() {
    return createMLflowRoutePath('/gateway/:endpointId');
  }

  static get editEndpointPage() {
    return createMLflowRoutePath('/gateway/:endpointId/edit');
  }
}

class GatewayRoutes {
  static get gatewayPageRoute() {
    return GatewayRoutePaths.gatewayPage;
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
