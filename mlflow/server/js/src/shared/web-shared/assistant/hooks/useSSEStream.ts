/**
 * Hook for managing SSE stream connections.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { createEventSource } from '../AssistantService';

interface UseSSEStreamOptions {
  onMessage?: (text: string) => void;
  onError?: (error: string) => void;
  onDone?: () => void;
  onStatus?: (status: string) => void;
}

interface UseSSEStreamReturn {
  isStreaming: boolean;
  error: string | null;
  connect: (sessionId: string) => void;
  disconnect: () => void;
}

export const useSSEStream = (options: UseSSEStreamOptions = {}): UseSSEStreamReturn => {
  const { onMessage, onError, onDone, onStatus } = options;
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const disconnect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  const connect = useCallback(
    (sessionId: string) => {
      // Close any existing connection
      disconnect();
      setError(null);
      setIsStreaming(true);

      const eventSource = createEventSource(sessionId);
      eventSourceRef.current = eventSource;

      // Inline event handling
      eventSource.addEventListener('message', (event) => {
        try {
          const data = JSON.parse(event.data);
          if ('text' in data) {
            onMessage?.(data.text);
          }
        } catch {
          onMessage?.(event.data);
        }
      });

      eventSource.addEventListener('status', (event) => {
        try {
          const data = JSON.parse((event as MessageEvent).data);
          if ('status' in data && data.status !== 'complete') {
            onStatus?.(data.status);
          }
        } catch {
          // Ignore parse errors for status events
        }
      });

      eventSource.addEventListener('done', () => {
        setIsStreaming(false);
        onDone?.();
        eventSource.close();
      });

      eventSource.addEventListener('error', (event) => {
        if (eventSource.readyState === EventSource.CLOSED) {
          return;
        }
        try {
          const data = JSON.parse((event as MessageEvent).data);
          const errorMsg = data.error || 'Unknown error';
          setError(errorMsg);
          setIsStreaming(false);
          onError?.(errorMsg);
        } catch {
          const errorMsg = 'Connection error';
          setError(errorMsg);
          setIsStreaming(false);
          onError?.(errorMsg);
        }
        eventSource.close();
      });
    },
    [disconnect, onMessage, onError, onDone, onStatus],
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    isStreaming,
    error,
    connect,
    disconnect,
  };
};
