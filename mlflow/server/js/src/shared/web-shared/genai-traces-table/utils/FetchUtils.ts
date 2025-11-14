import {
  fetchAPI,
  getAjaxUrl as workspaceAwareGetAjaxUrl,
  getDefaultHeaders as workspaceAwareGetDefaultHeaders,
  getDefaultHeadersFromCookies as workspaceAwareGetDefaultHeadersFromCookies,
} from '@mlflow/mlflow/src/common/utils/FetchUtils';

// eslint-disable-next-line no-restricted-globals
export const fetchFn = fetch; // use global fetch for oss

// Re-export fetchAPI for use in hooks - it already handles workspace headers
export { fetchAPI };

export const makeRequest = async <T>(path: string, method: 'POST' | 'GET', body?: T, signal?: AbortSignal) => {
  return fetchAPI(path, { method, body, signal });
};

export const getAjaxUrl = (relativeUrl: any) => workspaceAwareGetAjaxUrl(relativeUrl);
export const getDefaultHeadersFromCookies = workspaceAwareGetDefaultHeadersFromCookies;
export const getDefaultHeaders = workspaceAwareGetDefaultHeaders;
