import { getDefaultHeaders } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { matchPredefinedError } from '../../errors';

// eslint-disable-next-line no-restricted-globals
export const fetchFn = fetch; // use global fetch for oss

export const makeRequest = async <T>(path: string, method: 'POST' | 'GET', body?: T) => {
  const headers = {
    ...(body ? { 'Content-Type': 'application/json' } : {}),
    ...getDefaultHeaders(document.cookie),
  };
  const options: RequestInit = {
    method,
    headers,
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
