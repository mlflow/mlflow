import { createMLflowRoutePath, generatePath } from '../common/utils/RoutingUtils';

export enum GatewayPageId {
  gatewayPage = 'mlflow.gateway',
}

export class GatewayRoutePaths {
  static get gatewayPage() {
    return createMLflowRoutePath('/gateway');
  }
}

class GatewayRoutes {
  static get gatewayPageRoute() {
    return GatewayRoutePaths.gatewayPage;
  }
}

export default GatewayRoutes;
