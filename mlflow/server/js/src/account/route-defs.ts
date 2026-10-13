import type { DocumentTitleHandle } from '../common/utils/RoutingUtils';
import { createLazyRouteElement } from '../common/utils/RoutingUtils';
import { AccountPageId, AccountRoutePaths } from './routes';

export const getAccountRouteDefs = () => {
  return [
    {
      path: AccountRoutePaths.accountPage,
      element: createLazyRouteElement(() => import('./AccountPage')),
      pageId: AccountPageId.accountPage,
      handle: { getPageTitle: () => 'Account' } satisfies DocumentTitleHandle,
    },
  ];
};
