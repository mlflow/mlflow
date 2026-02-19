import { createMLflowRoutePath, generatePath } from '../common/utils/RoutingUtils';

// Route path definitions (used in defining route elements)
// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
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
// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
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
  static getCompareModelVersionsPageRoute(modelName: string, runsToVersions: Record<string, string>) {
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
