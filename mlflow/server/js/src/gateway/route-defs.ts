import { createLazyRouteElement } from '../common/utils/RoutingUtils';
import { GatewayPageId, GatewayRoutePaths } from './routes';

export const getGatewayRouteDefs = () => {
  return [
    {
      path: GatewayRoutePaths.gatewayPage,
      element: createLazyRouteElement(() => import('./pages/GatewayPage')),
      pageId: GatewayPageId.gatewayPage,
      handle: { title: 'AI Gateway' },
      children: [
        {
          path: 'api-keys',
          element: createLazyRouteElement(() => import('./pages/ApiKeysPage')),
          pageId: GatewayPageId.apiKeysPage,
          handle: { title: 'API Keys' },
        },
        {
          path: 'endpoints/create',
          element: createLazyRouteElement(() => import('./pages/CreateEndpointPage')),
          pageId: GatewayPageId.createEndpointPage,
          handle: { title: 'Create Endpoint' },
        },
        {
          path: 'endpoints/:endpointId',
          element: createLazyRouteElement(() => import('./pages/EndpointPage')),
          pageId: GatewayPageId.endpointDetailsPage,
          handle: { title: 'Endpoint Details' },
        },
      ],
    },
  ];
};
