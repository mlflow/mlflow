import { createLazyRouteElement } from '../common/utils/RoutingUtils';
import { SECRETS_BASE_ROUTE } from './routes';

export const getSecretsRouteDefs = () => {
  return [
    {
      path: SECRETS_BASE_ROUTE,
      element: createLazyRouteElement(() => import('./components/RoutesPage')),
      pageId: 'mlflow.routes',
    },
  ];
};
