import { createMLflowRoutePath, generatePath } from '../common/utils/RoutingUtils';

export enum GatewayPageId {
  gatewayPage = 'mlflow.gateway',
  apiKeysPage = 'mlflow.gateway.api-keys',
  usagePage = 'mlflow.gateway.usage',
  budgetsPage = 'mlflow.gateway.budgets',
  createEndpointPage = 'mlflow.gateway.create',
  endpointDetailsPage = 'mlflow.gateway.endpoint-details',
}

// following same pattern as other routes files
// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
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

  static get budgetsPage() {
    return createMLflowRoutePath('/gateway/budgets');
  }

  static get createEndpointPage() {
    return createMLflowRoutePath('/gateway/endpoints/create');
  }

  static get endpointDetailsPage() {
    return createMLflowRoutePath('/gateway/endpoints/:endpointId');
  }
}

// following same pattern as other routes files
// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
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

  static get budgetsPageRoute() {
    return GatewayRoutePaths.budgetsPage;
  }

  static get createEndpointPageRoute() {
    return GatewayRoutePaths.createEndpointPage;
  }

  static getEndpointDetailsRoute(endpointId: string, options?: { tab?: string; startTime?: string; endTime?: string }) {
    const path = generatePath(GatewayRoutePaths.endpointDetailsPage, { endpointId });
    if (!options) return path;
    const params = new URLSearchParams();
    if (options.tab) params.set('tab', options.tab);
    if (options.startTime && options.endTime) {
      params.set('startTimeLabel', 'CUSTOM');
      params.set('startTime', options.startTime);
      params.set('endTime', options.endTime);
    }
    const query = params.toString();
    return query ? `${path}?${query}` : path;
  }
}

export default GatewayRoutes;
