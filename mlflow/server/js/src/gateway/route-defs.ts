import { createLazyRouteElement } from '../common/utils/RoutingUtils';
import { GatewayPageId, GatewayRoutePaths } from './routes';

export const getGatewayRouteDefs = () => {
  return [
    {
      path: GatewayRoutePaths.gatewayPage,
      element: createLazyRouteElement(() => import('./pages/GatewayLayout')),
      pageId: GatewayPageId.gatewayPage,
      children: [
        {
          index: true,
          element: createLazyRouteElement(() => import('./pages/GatewayPage')),
          pageId: GatewayPageId.gatewayPage,
        },
        {
          path: 'api-keys',
          element: createLazyRouteElement(() => import('./pages/ApiKeysPage')),
          pageId: GatewayPageId.apiKeysPage,
        },
        {
          path: 'models',
          element: createLazyRouteElement(() => import('./pages/ModelDefinitionsPage')),
          pageId: GatewayPageId.modelDefinitionsPage,
        },
        {
          path: 'models/:modelDefinitionId',
          element: createLazyRouteElement(() => import('./pages/ModelDefinitionDetailsPage')),
          pageId: GatewayPageId.modelDefinitionDetailsPage,
        },
        {
          path: 'models/:modelDefinitionId/edit',
          element: createLazyRouteElement(() => import('./pages/EditModelDefinitionPage')),
          pageId: GatewayPageId.editModelDefinitionPage,
        },
        {
          path: 'endpoints/create',
          element: createLazyRouteElement(() => import('./pages/CreateEndpointPage')),
          pageId: GatewayPageId.createEndpointPage,
        },
        {
          path: 'endpoints/:endpointId',
          element: createLazyRouteElement(() => import('./pages/EndpointDetailsPage')),
          pageId: GatewayPageId.endpointDetailsPage,
        },
        {
          path: 'endpoints/:endpointId/edit',
          element: createLazyRouteElement(() => import('./pages/EditEndpointPage')),
          pageId: GatewayPageId.editEndpointPage,
        },
      ],
    },
  ];
};
