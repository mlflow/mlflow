import { JSONBig } from '../core/utils/json';

/**
 * Get the request headers for the given token.
 *
 * @param token - The token to use to authenticate the request.
 * @returns The request headers.
 */
export function getRequestHeaders(token?: string): Record<string, string> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  return headers;
}

/**
 * Make a request to the given URL with the given method, headers, body, and timeout.
 *
 * TODO: add retry logic for transient errors
 */
export async function makeRequest<T>(
  method: string,
  url: string,
  headers: Record<string, string>,
  body?: any,
  timeout?: number,
): Promise<T> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout ?? getDefaultTimeout());

  try {
    const response = await fetch(url, {
      method,
      headers: headers,
      body: body ? JSONBig.stringify(body) : undefined,
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      await handleErrorResponse(response);
    }

    // Handle empty responses (like DELETE operations)
    if (response.status === 204 || response.headers.get('content-length') === '0') {
      return {} as T;
    }

    const responseText = await response.text();
    return JSONBig.parse(responseText) as T;
  } catch (error) {
    clearTimeout(timeoutId);

    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        throw new Error(`Request timeout after ${timeout}ms`);
      }
      throw new Error(`API request failed: ${error.message}`);
    }
    throw new Error(`API request failed: ${String(error)}`);
  }
}

export async function handleErrorResponse(response: Response): Promise<void> {

let errorMessage = `HTTP ${response.status}: ${response.statusText}`;

try {
  const contentType = response.headers.get('content-type');

  if (contentType?.includes('application/json')) {
    const errorText = await response.text();
    const errorBody = JSONBig.parse(errorText) as { message?: string; error_code?: string };
    if (errorBody.message) {
      errorMessage = errorBody.message;
    } else if (errorBody.error_code) {
      errorMessage = `${errorBody.error_code}: ${errorBody.message || 'Unknown error'}`;
    }
  } else {
    // Not JSON, get first 200 chars of text for debugging
    const errorText = await response.text();
    console.debug(`Non-JSON error response: ${errorText.substring(0, 200)}...`);
    errorMessage = `${errorMessage} (received ${contentType || 'unknown'} instead of JSON)`;
  }
} catch (parseError) {
  console.debug(`Failed to parse error response: ${String(parseError)}`);
}

}

function getDefaultTimeout(): number {
  const envTimeout = process.env.MLFLOW_HTTP_REQUEST_TIMEOUT;
  if (envTimeout) {
    const timeout = parseInt(envTimeout, 10);
    if (!isNaN(timeout) && timeout > 0) {
      return timeout;
    }
  }
  return 30000; // Default 30 seconds
}
