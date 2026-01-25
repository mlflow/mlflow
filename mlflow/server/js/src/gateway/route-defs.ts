import { createLazyRouteElement, DocumentTitleHandle } from '../common/utils/RoutingUtils';
import { GatewayPageId, GatewayRoutePaths } from './routes';

export const getGatewayRouteDefs = () => {
  return [
    {
      path: GatewayRoutePaths.gatewayPage,
      element: createLazyRouteElement(() => import('./pages/GatewayPage')),
      pageId: GatewayPageId.gatewayPage,
      handle: { getPageTitle: () => 'AI Gateway' } satisfies DocumentTitleHandle,
      children: [
        {
          path: 'api-keys',
          element: createLazyRouteElement(() => import('./pages/ApiKeysPage')),
          pageId: GatewayPageId.apiKeysPage,
          handle: { getPageTitle: () => 'API Keys' } satisfies DocumentTitleHandle,
        },
        {
          path: 'usage',
          element: createLazyRouteElement(() => import('./pages/GatewayUsagePage')),
          pageId: GatewayPageId.usagePage,
          handle: { getPageTitle: () => 'Usage' } satisfies DocumentTitleHandle,
        },
        {
          path: 'endpoints/create',
          element: createLazyRouteElement(() => import('./pages/CreateEndpointPage')),
          pageId: GatewayPageId.createEndpointPage,
          handle: { getPageTitle: () => 'Create Endpoint' } satisfies DocumentTitleHandle,
        },
        {
          path: 'endpoints/:endpointId',
          element: createLazyRouteElement(() => import('./pages/EndpointPage')),
          pageId: GatewayPageId.endpointDetailsPage,
          handle: { getPageTitle: (params) => `Endpoint ${params['endpointId']}` } satisfies DocumentTitleHandle,
        },
      ],
    },
  ];
};
