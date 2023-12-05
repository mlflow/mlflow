import {
  createRouteElement,
  createMLflowRoutePath,
  generatePath,
} from '../common/utils/RoutingUtils';

import { CompareModelVersionsPage } from './components/CompareModelVersionsPage';
import { ModelListPage } from './components/ModelListPage';
import { ModelPage } from './components/ModelPage';
import { ModelVersionPage } from './components/ModelVersionPage';

// Route path definitions (used in defining route elements)
export class ModelRegistryRoutePaths {
  static get modelListPage() {
    return createMLflowRoutePath('/models');
  }
  static get modelPage() {
    return createMLflowRoutePath('/models/:modelName');
  }
  static get modelSubpage() {
    return createMLflowRoutePath('/models/:modelName/:subpage');
  }
  static get modelSubpageRouteWithName() {
    return createMLflowRoutePath('/models/:modelName/:subpage/:name');
  }
  static get modelVersionPage() {
    return createMLflowRoutePath('/models/:modelName/versions/:version');
  }
  static get compareModelVersionsPage() {
    return createMLflowRoutePath('/compare-model-versions');
  }
  static get createModel() {
    return createMLflowRoutePath('/createModel');
  }
}

// Concrete routes and functions for generating parametrized paths
export class ModelRegistryRoutes {
  static get modelListPageRoute() {
    return ModelRegistryRoutePaths.modelListPage;
  }
  static getModelPageRoute(modelName: string) {
    return generatePath(ModelRegistryRoutePaths.modelPage, {
      modelName: encodeURIComponent(modelName),
    });
  }
  static getModelPageServingRoute(modelName: string) {
    return generatePath(ModelRegistryRoutePaths.modelSubpage, {
      modelName: encodeURIComponent(modelName),
      subpage: PANES.SERVING,
    });
  }
  static getModelVersionPageRoute(modelName: string, version: string) {
    return generatePath(ModelRegistryRoutePaths.modelVersionPage, {
      modelName: encodeURIComponent(modelName),
      version,
    });
  }
  static getCompareModelVersionsPageRoute(
    modelName: string,
    runsToVersions: Record<string, string>,
  ) {
    const path = generatePath(ModelRegistryRoutePaths.compareModelVersionsPage);
    const query =
      `?name=${JSON.stringify(encodeURIComponent(modelName))}` +
      `&runs=${JSON.stringify(runsToVersions, (_, v) => (v === undefined ? null : v))}`;

    return [path, query].join('');
  }
}

export const PANES = Object.freeze({
  DETAILS: 'details',
  SERVING: 'serving',
});

export const getRouteDefs = () => [
  {
    path: ModelRegistryRoutePaths.modelListPage,
    element: createRouteElement(ModelListPage),
    pageId: 'mlflow.model-registry.model-list',
  },
  {
    path: ModelRegistryRoutePaths.modelPage,
    element: createRouteElement(ModelPage),
    pageId: 'mlflow.model-registry.model-page',
  },
  {
    path: ModelRegistryRoutePaths.modelSubpage,
    element: createRouteElement(ModelPage),
    pageId: 'mlflow.model-registry.model-page.subpage',
  },
  {
    path: ModelRegistryRoutePaths.modelSubpageRouteWithName,
    element: createRouteElement(ModelPage),
    pageId: 'mlflow.model-registry.model-page.subpage.section',
  },
  {
    path: ModelRegistryRoutePaths.modelVersionPage,
    element: createRouteElement(ModelVersionPage),
    pageId: 'mlflow.model-registry.model-version-page',
  },
  {
    path: ModelRegistryRoutePaths.compareModelVersionsPage,
    element: createRouteElement(CompareModelVersionsPage),
    pageId: 'mlflow.model-registry.compare-model-versions',
  },
];
