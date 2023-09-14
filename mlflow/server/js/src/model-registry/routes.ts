/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

export const modelListPageRoute = '/models';
export const modelPageRoute = '/models/:modelName';
export const modelSubpageRoute = '/models/:modelName/:subpage';
export const modelSubpageRouteWithName = '/models/:modelName/:subpage/:name';
export const modelVersionPageRoute = '/models/:modelName/versions/:version';
export const compareModelVersionsPageRoute = '/compare-model-versions';
export const getModelPageRoute = (modelName: any) => `/models/${encodeURIComponent(modelName)}`;
export const getModelVersionPageRoute = (modelName: any, version: any) =>
  `/models/${encodeURIComponent(modelName)}/versions/${version}`;
// replace undefined values with null, since undefined is not a valid JSON value
export const getCompareModelVersionsPageRoute = (modelName: any, runsToVersions: any) =>
  `/compare-model-versions?name=${JSON.stringify(encodeURIComponent(modelName))}` +
  `&runs=${JSON.stringify(runsToVersions, (k, v) => (v === undefined ? null : v))}`;
export const PANES = Object.freeze({
  DETAILS: 'details',
  SERVING: 'serving',
});

export const getRouteDefs = () => [
  // TODO(ML-33996): Add new model registry route definitions here
  // {
  //   path: createMLflowRoutePath('/models'),
  //   component: () => import('./components/ModelListPage'),
  //   pageId: 'mlflow.models.list',
  // },
];
