import { createLazyRouteElement } from '../common/utils/RoutingUtils';
import { GatewayPageId, GatewayRoutePaths } from './routes';

export const getGatewayRouteDefs = () => {
  return [
    {
      path: GatewayRoutePaths.gatewayPage,
      element: createLazyRouteElement(() => import('./pages/GatewayPage')),
      pageId: GatewayPageId.gatewayPage,
    },
  ];
};
