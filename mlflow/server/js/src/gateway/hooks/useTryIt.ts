import { useCallback } from 'react';
import { getDefaultHeaders } from '../../common/utils/FetchUtils';

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
      const response = await fetch(tryItRequestUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getDefaultHeaders(document.cookie),
        },
        body: JSON.stringify(parsed),
      });
      const text = await response.text();
      if (!response.ok) {
        setSendError(`Request failed (${response.status})`);
        setResponseBody(text || '');
        return;
      }
      try {
        const formatted = JSON.stringify(JSON.parse(text), null, 2);
        setResponseBody(formatted);
      } catch {
        setResponseBody(text);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setSendError(message);
      setResponseBody('');
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
