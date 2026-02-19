import { createLazyRouteElement, type DocumentTitleHandle } from '../common/utils/RoutingUtils';

import { ModelRegistryRoutePaths } from './routes';

export const getRouteDefs = () => [
  {
    path: ModelRegistryRoutePaths.modelListPage,
    element: createLazyRouteElement(() => import('./components/ModelListPageWrapper')),
    pageId: 'mlflow.model-registry.model-list',
    handle: { getPageTitle: () => 'Models' } satisfies DocumentTitleHandle,
  },
  {
    path: ModelRegistryRoutePaths.modelPage,
    element: createLazyRouteElement(() => import('./components/ModelPage')),
    pageId: 'mlflow.model-registry.model-page',
    handle: { getPageTitle: (params) => `Model: ${params['modelName']}` } satisfies DocumentTitleHandle,
  },
  {
    path: ModelRegistryRoutePaths.modelSubpage,
    element: createLazyRouteElement(() => import('./components/ModelPage')),
    pageId: 'mlflow.model-registry.model-page.subpage',
    handle: { getPageTitle: (params) => `Model: ${params['modelName']}` } satisfies DocumentTitleHandle,
  },
  {
    path: ModelRegistryRoutePaths.modelSubpageRouteWithName,
    element: createLazyRouteElement(() => import('./components/ModelPage')),
    pageId: 'mlflow.model-registry.model-page.subpage.section',
    handle: { getPageTitle: (params) => `Model: ${params['modelName']}` } satisfies DocumentTitleHandle,
  },
  {
    path: ModelRegistryRoutePaths.modelVersionPage,
    element: createLazyRouteElement(() => import('./components/ModelVersionPage')),
    pageId: 'mlflow.model-registry.model-version-page',
    handle: { getPageTitle: (params) => `${params['modelName']} v${params['version']}` } satisfies DocumentTitleHandle,
  },
  {
    path: ModelRegistryRoutePaths.compareModelVersionsPage,
    element: createLazyRouteElement(() => import('./components/CompareModelVersionsPage')),
    pageId: 'mlflow.model-registry.compare-model-versions',
    handle: { getPageTitle: () => 'Compare Model Versions' } satisfies DocumentTitleHandle,
  },
];
