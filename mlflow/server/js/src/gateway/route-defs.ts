import { createLazyRouteElement } from '../common/utils/RoutingUtils';
import { GatewayPageId, GatewayRoutePaths } from './routes';

export const getGatewayRouteDefs = () => {
  return [
    {
      path: GatewayRoutePaths.gatewayPage,
      element: createLazyRouteElement(() => import('./pages/GatewayPage')),
      pageId: GatewayPageId.gatewayPage,
      children: [
        {
          path: 'api-keys',
          element: createLazyRouteElement(() => import('./pages/ApiKeysPage')),
          pageId: GatewayPageId.apiKeysPage,
        },
      ],
    },
  ];
};
