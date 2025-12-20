import { matchPredefinedError } from '@databricks/web-shared/errors';
import {
  getAjaxUrl as workspaceAwareGetAjaxUrl,
  getDefaultHeaders as workspaceAwareGetDefaultHeaders,
  getDefaultHeadersFromCookies as workspaceAwareGetDefaultHeadersFromCookies,
} from '@mlflow/mlflow/src/common/utils/FetchUtils';

// eslint-disable-next-line no-restricted-globals -- See go/spog-fetch
const fetchFn = fetch;

function serializeRequestBody(payload: any | FormData | Blob) {
  if (payload === undefined) {
    return undefined;
  }
  return typeof payload === 'string' || payload instanceof FormData || payload instanceof Blob
    ? payload
    : JSON.stringify(payload);
}

export const fetchAPI = async (
  url: string,
  method: 'POST' | 'GET' | 'PATCH' | 'DELETE' = 'GET',
  body?: any,
  signal?: AbortSignal,
) => {
  const options: RequestInit = {
    method,
    signal,
    headers: {
      ...(body ? { 'Content-Type': 'application/json' } : {}),
      ...workspaceAwareGetDefaultHeaders(document.cookie),
    },
  };

  if (body) {
    options.body = serializeRequestBody(body);
  }
  const response = await fetchFn(url, options);

  if (!response.ok) {
    const predefinedError = matchPredefinedError(response);
    if (predefinedError) {
      try {
        // Attempt to use message from the response
        const message = (await response.json()).message;
        predefinedError.message = message ?? predefinedError.message;
      } catch {
        // If the message can't be parsed, use default one
      }
      throw predefinedError;
    }
  }
  return response.json();
};

export const getAjaxUrl = (relativeUrl: any) => workspaceAwareGetAjaxUrl(relativeUrl);
export const getDefaultHeadersFromCookies = workspaceAwareGetDefaultHeadersFromCookies;
export const getDefaultHeaders = workspaceAwareGetDefaultHeaders;
