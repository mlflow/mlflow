export const modelListPageRoute = '/models';
export const modelPageRoute = '/models/:modelName';
export const modelSubpageRoute = '/models/:modelName/:subpage';
export const modelVersionPageRoute = '/models/:modelName/versions/:version';
export const compareModelVersionsPageRoute = '/compare-model-versions';
export const getModelPageRoute = (modelName) => `/models/${encodeURIComponent(modelName)}`;
export const getModelVersionPageRoute = (modelName, version) =>
  `/models/${encodeURIComponent(modelName)}/versions/${version}`;
// replace undefined values with null, since undefined is not a valid JSON value
export const getCompareModelVersionsPageRoute = (modelName, runsToVersions) =>
  `/compare-model-versions?name=${JSON.stringify(encodeURIComponent(modelName))}` +
  `&runs=${JSON.stringify(runsToVersions, (k, v) => (v === undefined ? null : v))}`;

export const getModelVersionPageURL = (modelName, version) => {
  const modelRoute = getModelVersionPageRoute(modelName, version);
  if (window.self !== window.top) {
    // If running in an iframe, include the parent params and assume mlflow served at #
    const parentHref = window.parent.location.href;
    const parentHrefBeforeMlflowHash = parentHref.split('#')[0];
    return `${parentHrefBeforeMlflowHash}#mlflow${modelRoute}`;
  }
  return `./#${modelRoute}`; // issue-2213 use relative path in case there is a url prefix
};
