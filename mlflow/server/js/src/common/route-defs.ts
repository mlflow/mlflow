import PageNotFoundView from './components/PageNotFoundView';
import { createMLflowRoutePath, createRouteElement } from './utils/RoutingUtils';
import type { DocumentTitleHandle } from './hooks/useDocumentTitle';

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
