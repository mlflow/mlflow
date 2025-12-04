import { createMLflowRoutePath, generatePath } from '../common/utils/RoutingUtils';

export enum GatewayPageId {
  gatewayPage = 'mlflow.gateway',
  apiKeysPage = 'mlflow.gateway.api-keys',
  modelDefinitionsPage = 'mlflow.gateway.model-definitions',
  modelDefinitionDetailsPage = 'mlflow.gateway.model-definitions.details',
  editModelDefinitionPage = 'mlflow.gateway.model-definitions.edit',
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

  static get modelDefinitionsPage() {
    return createMLflowRoutePath('/gateway/models');
  }

  static get modelDefinitionDetailsPage() {
    return createMLflowRoutePath('/gateway/models/:modelDefinitionId');
  }

  static get editModelDefinitionPage() {
    return createMLflowRoutePath('/gateway/models/:modelDefinitionId/edit');
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

  static get modelDefinitionsPageRoute() {
    return GatewayRoutePaths.modelDefinitionsPage;
  }

  static getModelDefinitionDetailsRoute(modelDefinitionId: string) {
    return generatePath(GatewayRoutePaths.modelDefinitionDetailsPage, { modelDefinitionId });
  }

  static getEditModelDefinitionRoute(modelDefinitionId: string) {
    return generatePath(GatewayRoutePaths.editModelDefinitionPage, { modelDefinitionId });
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
