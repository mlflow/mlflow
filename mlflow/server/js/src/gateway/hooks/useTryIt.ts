import { useCallback } from 'react';
import { fetchOrFail } from '../../common/utils/FetchUtils';
import { NetworkRequestError } from '../../shared/web-shared/errors/PredefinedErrors';

function formatResponseText(text: string): string {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

export interface UseTryItParams {
  requestBody: string;
  tryItRequestUrl: string;
  onSuccess?: (formattedResponse: string) => void;
  onError?: (error: Error, responseText?: string) => void;
  onSendingChange?: (sending: boolean) => void;
  onReset?: () => void;
}

export function useTryIt({
  requestBody,
  tryItRequestUrl,
  onSuccess,
  onError,
  onSendingChange,
  onReset,
}: UseTryItParams) {
  const handleSendRequest = useCallback(async () => {
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(requestBody);
    } catch {
      onError?.(new Error('Invalid JSON in request body'));
      return;
    }

    onSendingChange?.(true);
    try {
      const response = await fetchOrFail(tryItRequestUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(parsed),
      });
      const text = await response.text();
      onSuccess?.(formatResponseText(text));
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      const rawText = err instanceof NetworkRequestError ? (await err.response?.text()) || '' : undefined;
      const responseText = rawText !== undefined ? formatResponseText(rawText) : undefined;
      onError?.(error, responseText);
    } finally {
      onSendingChange?.(false);
    }
  }, [requestBody, tryItRequestUrl, onSuccess, onError, onSendingChange]);

  const handleResetExample = useCallback(() => {
    onReset?.();
  }, [onReset]);

  return { handleSendRequest, handleResetExample };
}
