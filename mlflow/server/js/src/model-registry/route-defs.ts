import { createLazyRouteElement } from '../common/utils/RoutingUtils';

import { ModelRegistryRoutePaths } from './routes';

export const getRouteDefs = () => [
  {
    path: ModelRegistryRoutePaths.modelListPage,
    element: createLazyRouteElement(() => import('./components/ModelListPageWrapper')),
    pageId: 'mlflow.model-registry.model-list',
    handle: { title: 'Models' },
  },
  {
    path: ModelRegistryRoutePaths.modelPage,
    element: createLazyRouteElement(() => import('./components/ModelPage')),
    pageId: 'mlflow.model-registry.model-page',
    handle: { title: 'Model' },
  },
  {
    path: ModelRegistryRoutePaths.modelSubpage,
    element: createLazyRouteElement(() => import('./components/ModelPage')),
    pageId: 'mlflow.model-registry.model-page.subpage',
    handle: { title: 'Model' },
  },
  {
    path: ModelRegistryRoutePaths.modelSubpageRouteWithName,
    element: createLazyRouteElement(() => import('./components/ModelPage')),
    pageId: 'mlflow.model-registry.model-page.subpage.section',
    handle: { title: 'Model' },
  },
  {
    path: ModelRegistryRoutePaths.modelVersionPage,
    element: createLazyRouteElement(() => import('./components/ModelVersionPage')),
    pageId: 'mlflow.model-registry.model-version-page',
    handle: { title: 'Model Version' },
  },
  {
    path: ModelRegistryRoutePaths.compareModelVersionsPage,
    element: createLazyRouteElement(() => import('./components/CompareModelVersionsPage')),
    pageId: 'mlflow.model-registry.compare-model-versions',
    handle: { title: 'Compare Model Versions' },
  },
];
