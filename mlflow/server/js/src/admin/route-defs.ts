import type { DocumentTitleHandle } from '../common/utils/RoutingUtils';
import { createLazyRouteElement } from '../common/utils/RoutingUtils';
import { AdminPageId, AdminRoutePaths } from './routes';

export const getAdminRouteDefs = () => {
  return [
    {
      path: AdminRoutePaths.adminPage,
      element: createLazyRouteElement(() => import('./pages/AdminPage')),
      pageId: AdminPageId.adminPage,
      handle: { getPageTitle: () => 'Platform Admin' } satisfies DocumentTitleHandle,
    },
    {
      path: AdminRoutePaths.roleDetailPage,
      element: createLazyRouteElement(() => import('./pages/RoleDetailPage')),
      pageId: AdminPageId.roleDetailPage,
      handle: { getPageTitle: (params) => `Role ${params['roleId']}` } satisfies DocumentTitleHandle,
    },
    {
      path: AdminRoutePaths.userDetailPage,
      element: createLazyRouteElement(() => import('./pages/UserDetailPage')),
      pageId: AdminPageId.userDetailPage,
      handle: { getPageTitle: (params) => `User ${params['username']}` } satisfies DocumentTitleHandle,
    },
  ];
};
