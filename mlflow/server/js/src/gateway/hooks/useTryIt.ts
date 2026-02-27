import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { fetchOrFail } from '../../common/utils/FetchUtils';
import { NetworkRequestError } from '../../shared/web-shared/errors/PredefinedErrors';

function formatResponseText(text: string): string {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

interface TryItError extends Error {
  responseBody?: string;
}

/** Creates a TryItError with an optional response body for the UI to display. */
function createTryItError(message: string, responseBody?: string): TryItError {
  const err = new Error(message) as TryItError;
  err.responseBody = responseBody;
  return err;
}

/** Try-it fetch options. Only headers and signal are allowed; method and body are set by the hook. */
export type TryItFetchOptions = Pick<RequestInit, 'headers' | 'signal'>;

export interface UseTryItParams {
  tryItRequestUrl: string;
  /** Optional fetch options (headers, signal) passed through to fetchOrFail. Method and body are set by the hook. */
  options?: TryItFetchOptions;
}

/**
 * Hook to send a "Try it" request to the gateway.
 *
 * @example
 * const { data, isLoading, error, sendRequest, reset } = useTryIt({ tryItRequestUrl });
 * // data = last successful response (formatted); error = last error (with error.responseBody if present)
 */
export function useTryIt({ tryItRequestUrl, options }: UseTryItParams) {
  const mutation = useMutation<string, TryItError, string>({
    mutationFn: async (requestBody: string) => {
      let parsed: Record<string, unknown>;
      try {
        parsed = JSON.parse(requestBody);
      } catch {
        throw createTryItError('Invalid JSON in request body');
      }

      try {
        const requestHeaders =
          options?.headers && typeof options.headers === 'object' && !(options.headers instanceof Headers)
            ? (options.headers as Record<string, string>)
            : {};
        const response = await fetchOrFail(tryItRequestUrl, {
          ...options,
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...requestHeaders,
          },
          body: JSON.stringify(parsed),
        });
        const text = await response.text();
        return formatResponseText(text);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        const rawText = err instanceof NetworkRequestError ? (await err.response?.text()) || '' : undefined;
        const responseBody = rawText !== undefined ? formatResponseText(rawText) : undefined;
        throw createTryItError(message, responseBody);
      }
    },
  });

  return {
    data: mutation.data,
    isLoading: mutation.isLoading,
    error: mutation.error ?? undefined,
    sendRequest: mutation.mutate,
    reset: mutation.reset,
  };
}
