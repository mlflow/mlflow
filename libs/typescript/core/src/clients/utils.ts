import { AuthProvider } from '../auth/types';
import { JSONBig } from '../core/utils/json';

/**
 * Get the request headers for the given token or basic auth credentials.
 * Token will be used if provided, otherwise basic auth credentials will be used.
 *
 * @param token - The token to use to authenticate the request.
 * @param username - The username to use to authenticate the request with basic auth.
 * @param password - The password to use to authenticate the request with basic auth.
 * @returns The request headers.
 */
export function getRequestHeaders(
  token?: string,
  username?: string,
  password?: string
): Record<string, string> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  } else if (username && password) {
    headers['Authorization'] = `Basic ${Buffer.from(`${username}:${password}`).toString('base64')}`;
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
  timeout?: number
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
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
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

/**
 * Make an authenticated request using the provided auth provider.
 * The auth provider handles token refresh automatically.
 *
 * @param method - The HTTP method to use.
 * @param url - The URL to send the request to.
 * @param authProvider - The authentication provider to use.
 * @param body - Optional request body.
 * @param timeout - Optional timeout in milliseconds.
 * @returns The parsed response body.
 */
export async function makeAuthenticatedRequest<T>(
  method: string,
  url: string,
  authProvider: AuthProvider,
  body?: any,
  timeout?: number
): Promise<T> {
  // Get fresh auth credentials (provider handles refresh if needed)
  const authResult = await authProvider.authenticate();

  const headers: Record<string, string> = {
    'Content-Type': 'application/json'
  };

  if (authResult.authorizationHeader) {
    headers['Authorization'] = authResult.authorizationHeader;
  }

  return makeRequest<T>(method, url, headers, body, timeout);
}
