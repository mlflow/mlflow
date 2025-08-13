import { matchPredefinedError } from '@databricks/web-shared/errors';
import { getDefaultHeaders } from '../../../common/utils/FetchUtils';

function serializeRequestBody(payload: any | FormData | Blob) {
  if (payload === undefined) {
    return undefined;
  }
  return typeof payload === 'string' || payload instanceof FormData || payload instanceof Blob
    ? payload
    : JSON.stringify(payload);
}

// Helper method to make a request to the backend.
export const loggedModelsDataRequest = async (
  url: string,
  method: 'POST' | 'GET' | 'PATCH' | 'DELETE' = 'GET',
  body?: any,
) => {
  const headers = {
    ...(body ? { 'Content-Type': 'application/json' } : {}),
    ...getDefaultHeaders(document.cookie),
  };
  const response = await fetch(url, {
    method,
    body: serializeRequestBody(body),
    headers,
  });
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
