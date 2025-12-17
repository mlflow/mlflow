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
    {
      path: GatewayRoutePaths.createEndpointPage,
      element: createLazyRouteElement(() => import('./pages/CreateEndpointPage')),
      pageId: GatewayPageId.createEndpointPage,
    },
    {
      path: GatewayRoutePaths.endpointDetailsPage,
      element: createLazyRouteElement(() => import('./pages/EndpointDetailsPage')),
      pageId: GatewayPageId.endpointDetailsPage,
    },
    {
      path: GatewayRoutePaths.editEndpointPage,
      element: createLazyRouteElement(() => import('./pages/EditEndpointPage')),
      pageId: GatewayPageId.editEndpointPage,
    },
  ];
};
