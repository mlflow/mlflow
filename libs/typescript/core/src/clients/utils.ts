import { JSONBig } from '../core/utils/json';
import { HeadersProvider } from '../auth';

const MAX_ERROR_BODY_LENGTH = 1000;

/**
 * Error thrown when an HTTP request to the MLflow backend returns a non-2xx response.
 * Carries the parsed status code and (when present) the `error_code` field from the
 * response body so callers can branch on status without matching error messages.
 */
export class MlflowHttpError extends Error {
  readonly status: number;
  readonly statusText: string;
  readonly body: string;
  readonly errorCode?: string;

  constructor(status: number, statusText: string, body: string) {
    super(MlflowHttpError.formatMessage(status, statusText, body));
    this.name = 'MlflowHttpError';
    this.status = status;
    this.statusText = statusText;
    this.body = body;
    this.errorCode = MlflowHttpError.extractErrorCode(body);
  }

  private static formatMessage(status: number, statusText: string, body: string): string {
    let message = `HTTP ${status}: ${statusText}`;
    if (body) {
      message +=
        body.length > MAX_ERROR_BODY_LENGTH
          ? ` - ${body.substring(0, MAX_ERROR_BODY_LENGTH)}... (truncated)`
          : ` - ${body}`;
    }
    return message;
  }

  private static extractErrorCode(body: string): string | undefined {
    if (!body) {
      return undefined;
    }
    try {
      const parsed: unknown = JSON.parse(body);
      if (
        parsed &&
        typeof parsed === 'object' &&
        'error_code' in parsed &&
        typeof (parsed as { error_code: unknown }).error_code === 'string'
      ) {
        return (parsed as { error_code: string }).error_code;
      }
    } catch {
      // body is not JSON
    }
    return undefined;
  }
}

/**
 * Make a request to the given URL with the given method, headers, body, and timeout.
 *
 * TODO: add retry logic for transient errors
 */
export async function makeRequest<T>(
  method: string,
  url: string,
  headerProvider: HeadersProvider,
  body?: any,
  timeout?: number,
): Promise<T> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout ?? getDefaultTimeout());
  const headers = await headerProvider();

  try {
    const response = await fetch(url, {
      method,
      headers: headers,
      body: body ? JSONBig.stringify(body) : undefined,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      let responseBody = '';
      try {
        responseBody = await response.text();
      } catch {
        // If we can't read the body, leave it empty
      }
      throw new MlflowHttpError(response.status, response.statusText, responseBody);
    }

    // Handle empty responses (like DELETE operations)
    if (response.status === 204 || response.headers.get('content-length') === '0') {
      return {} as T;
    }

    const responseText = await response.text();
    return JSONBig.parse(responseText) as T;
  } catch (error) {
    clearTimeout(timeoutId);

    if (error instanceof MlflowHttpError) {
      throw error;
    }

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
