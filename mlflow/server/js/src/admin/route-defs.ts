import type { DocumentTitleHandle } from '../common/utils/RoutingUtils';
import { createLazyRouteElement } from '../common/utils/RoutingUtils';
import { AdminPageId, AdminRoutePaths } from './routes';

export const getAdminRouteDefs = () => {
  return [
    {
      path: AdminRoutePaths.accountPage,
      element: createLazyRouteElement(() => import('./pages/AccountPage')),
      pageId: AdminPageId.accountPage,
      handle: { getPageTitle: () => 'Account' } satisfies DocumentTitleHandle,
    },
  ];
};
