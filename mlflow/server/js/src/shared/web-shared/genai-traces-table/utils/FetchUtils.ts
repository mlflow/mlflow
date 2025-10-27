import { getDefaultHeaders } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { matchPredefinedError } from '../../errors';

// eslint-disable-next-line no-restricted-globals
export const fetchFn = fetch; // use global fetch for oss

export const makeRequest = async <T>(path: string, method: 'POST' | 'GET', body?: T) => {
  const options: RequestInit = {
    method,
    headers: {
      ...(body ? { 'Content-Type': 'application/json' } : {}),
      ...getDefaultHeaders(document.cookie),
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

export const getAjaxUrl = (relativeUrl: any) => {
  if (process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] === 'true' && !relativeUrl.startsWith('/')) {
    return '/' + relativeUrl;
  }
  return relativeUrl;
};
