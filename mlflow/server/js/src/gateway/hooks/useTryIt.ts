import { useCallback } from 'react';
import { fetchOrFail } from '../../common/utils/FetchUtils';
import { NetworkRequestError } from '../../shared/web-shared/errors/PredefinedErrors';

export interface UseTryItParams {
  requestBody: string;
  tryItRequestUrl: string;
  tryItDefaultBody: string;
  setRequestBody: (value: string) => void;
  setResponseBody: (value: string) => void;
  setSendError: (value: string | null) => void;
  setIsSending: (value: boolean) => void;
}

export function useTryIt({
  requestBody,
  tryItRequestUrl,
  tryItDefaultBody,
  setRequestBody,
  setResponseBody,
  setSendError,
  setIsSending,
}: UseTryItParams) {
  const handleSendRequest = useCallback(async () => {
    const setResponseFromText = (text: string) => {
      try {
        setResponseBody(JSON.stringify(JSON.parse(text), null, 2));
      } catch {
        setResponseBody(text);
      }
    };

    setSendError(null);
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(requestBody);
    } catch {
      setSendError('Invalid JSON in request body');
      setResponseBody('');
      return;
    }
    setIsSending(true);
    setResponseBody('');
    try {
      const response = await fetchOrFail(tryItRequestUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(parsed),
      });
      const text = await response.text();
      setResponseFromText(text);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      const text = err instanceof NetworkRequestError ? (await err.response?.text()) || '' : '';
      setSendError(message);
      setResponseFromText(text);
    } finally {
      setIsSending(false);
    }
  }, [requestBody, tryItRequestUrl]);

  const handleResetExample = useCallback(() => {
    setRequestBody(tryItDefaultBody);
    setResponseBody('');
    setSendError(null);
  }, [tryItDefaultBody]);

  return { handleSendRequest, handleResetExample };
}
