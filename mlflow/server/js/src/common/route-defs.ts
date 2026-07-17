import PageNotFoundView from './components/PageNotFoundView';
import { createMLflowRoutePath, createRouteElement, type DocumentTitleHandle } from './utils/RoutingUtils';

/**
 * Common route definitions. For the time being it's 404 page only.
 */
export const getRouteDefs = () => [
  {
    path: createMLflowRoutePath('/*'),
    element: createRouteElement(PageNotFoundView),
    pageId: 'mlflow.common.not-found',
    handle: { getPageTitle: () => 'Page Not Found' } satisfies DocumentTitleHandle,
  },
];
