import type { DocumentTitleHandle } from '../common/utils/RoutingUtils';
import { createLazyRouteElement } from '../common/utils/RoutingUtils';
import { AdminPageId, AdminRoutePaths } from './routes';

export const getAdminRouteDefs = () => {
  return [
    {
      path: AdminRoutePaths.adminPage,
      element: createLazyRouteElement(() => import('./pages/AdminPage')),
      pageId: AdminPageId.adminPage,
      handle: { getPageTitle: () => 'Admin' } satisfies DocumentTitleHandle,
    },
    {
      path: AdminRoutePaths.roleDetailPage,
      element: createLazyRouteElement(() => import('./pages/RoleDetailPage')),
      pageId: AdminPageId.roleDetailPage,
      handle: { getPageTitle: (params) => `Role ${params['roleId']}` } satisfies DocumentTitleHandle,
    },
    {
      path: AdminRoutePaths.accountPage,
      element: createLazyRouteElement(() => import('./pages/AccountPage')),
      pageId: AdminPageId.accountPage,
      handle: { getPageTitle: () => 'Account' } satisfies DocumentTitleHandle,
    },
  ];
};
