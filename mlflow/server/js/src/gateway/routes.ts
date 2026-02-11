import { createMLflowRoutePath, generatePath } from '../common/utils/RoutingUtils';

export enum GatewayPageId {
  gatewayPage = 'mlflow.gateway',
  apiKeysPage = 'mlflow.gateway.api-keys',
  usagePage = 'mlflow.gateway.usage',
  createEndpointPage = 'mlflow.gateway.create',
  endpointDetailsPage = 'mlflow.gateway.endpoint-details',
}

export class GatewayRoutePaths {
  static get gatewayPage() {
    return createMLflowRoutePath('/gateway');
  }

  static get apiKeysPage() {
    return createMLflowRoutePath('/gateway/api-keys');
  }

  static get usagePage() {
    return createMLflowRoutePath('/gateway/usage');
  }

  static get createEndpointPage() {
    return createMLflowRoutePath('/gateway/endpoints/create');
  }

  static get endpointDetailsPage() {
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

  static get usagePageRoute() {
    return GatewayRoutePaths.usagePage;
  }

  static get createEndpointPageRoute() {
    return GatewayRoutePaths.createEndpointPage;
  }

  static getEndpointDetailsRoute(endpointId: string) {
    return generatePath(GatewayRoutePaths.endpointDetailsPage, { endpointId });
  }
}

export default GatewayRoutes;
