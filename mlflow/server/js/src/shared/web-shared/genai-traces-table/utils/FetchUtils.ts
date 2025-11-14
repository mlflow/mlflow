import {
  getAjaxUrl as workspaceAwareGetAjaxUrl,
  getDefaultHeaders as workspaceAwareGetDefaultHeaders,
  getDefaultHeadersFromCookies as workspaceAwareGetDefaultHeadersFromCookies,
} from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { matchPredefinedError } from '../../errors';

// eslint-disable-next-line no-restricted-globals
export const fetchFn = fetch; // use global fetch for oss

export const makeRequest = async <T>(path: string, method: 'POST' | 'GET', body?: T, signal?: AbortSignal) => {
  const options: RequestInit = {
    method,
    signal,
    headers: {
      ...(body ? { 'Content-Type': 'application/json' } : {}),
      ...workspaceAwareGetDefaultHeaders(document.cookie),
    },
  };

  if (body) {
    options.body = JSON.stringify(body);
  }
  const response = await fetchFn(path, options);

  if (!response.ok) {
    const error = matchPredefinedError(response);
    try {
      const errorMessageFromResponse = await (await response.json()).message;
      if (errorMessageFromResponse) {
        error.message = errorMessageFromResponse;
      }
    } catch {
      // do nothing
    }
    throw error;
  }

  return response.json();
};

export const getAjaxUrl = (relativeUrl: any) => workspaceAwareGetAjaxUrl(relativeUrl);
export const getDefaultHeadersFromCookies = workspaceAwareGetDefaultHeadersFromCookies;
export const getDefaultHeaders = workspaceAwareGetDefaultHeaders;
