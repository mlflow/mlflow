export const modelListPageRoute = '/models';
export const modelPageRoute = '/models/:modelName';
export const modelVersionPageRoute = '/models/:modelName/versions/:version';
export const compareModelVersionsPageRoute = '/compare-model-versions';
export const getModelPageRoute = (modelName) => `/models/${modelName}`;
export const getModelVersionPageRoute = (modelName, version) =>
  `/models/${modelName}/versions/${version}`;
export const getCompareModelVersionsPageRoute = (modelName, runsToVersions) =>
  `/compare-model-versions?name=${JSON.stringify(modelName)}` +
  `&runs=${JSON.stringify(runsToVersions)}`;

export const getModelVersionPageURL = (modelName, version) => {
  const modelRoute = getModelVersionPageRoute(modelName, version);
  if (window.self !== window.top) {
    // If running in an iframe, assume the UI is served at #mlflow/... and link to that
    const parentOrigin = window.parent.location.origin;
    return `${parentOrigin}/#mlflow${modelRoute}`;
  }
  return `./#${modelRoute}`; // issue-2213 use relative path in case there is a url prefix
};
