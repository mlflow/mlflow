/**
 * Service layer for Assistant Agent API calls.
 */

import type { MessageRequest } from './types';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';

const API_BASE = getAjaxUrl('ajax-api/3.0/mlflow/assistant');

/**
 * Create an EventSource for streaming responses.
 */
export const createEventSource = (sessionId: string): EventSource => {
  return new EventSource(`${API_BASE}/sessions/${sessionId}/stream`);
};

/**
 * Cancel an active session by terminating the backend process.
 */
export const cancelSession = async (sessionId: string): Promise<{ success: boolean; message: string }> => {
  const response = await fetch(`${API_BASE}/session/${sessionId}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ status: 'cancelled' }),
  });

  if (!response.ok) {
    throw new Error('Failed to cancel session');
  }

  return response.json();
};

export interface SendMessageStreamCallbacks {
  onMessage: (text: string) => void;
  onError: (error: string) => void;
  onDone: () => void;
  onStatus?: (status: string) => void;
  onSessionId?: (sessionId: string) => void;
  onInterrupted?: () => void;
}

export interface SendMessageStreamResult {
  eventSource: EventSource | null;
}

/**
 * Send a message and get the response stream via SSE.
 * First POSTs to /message to initiate, then connects to SSE endpoint.
 * Returns the EventSource so caller can close it if needed (e.g., on cancel).
 */
export const sendMessageStream = async (
  request: MessageRequest,
  callbacks: SendMessageStreamCallbacks,
): Promise<SendMessageStreamResult> => {
  const { onMessage, onError, onDone, onStatus, onSessionId, onInterrupted } = callbacks;

  try {
    // Step 1: POST the message to initiate processing
    const response = await fetch(`${API_BASE}/message`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.text();
      onError(`Failed to send message: ${error}`);
      return { eventSource: null };
    }

    // Step 2: Get the session_id from the response
    const result = await response.json();
    const sessionId = result.session_id;

    if (!sessionId) {
      onError('No session_id returned from server');
      return { eventSource: null };
    }

    // Notify caller of the session ID
    onSessionId?.(sessionId);

    // Step 3: Connect to the SSE endpoint to receive the stream
    const eventSource = createEventSource(sessionId);

    // Listen for 'message' events (contains assistant's response)
    eventSource.addEventListener('message', (event) => {
      try {
        const data = JSON.parse(event.data);
        // Backend sends: {"message": {"role": "assistant", "content": "..."}}
        if (data.message && data.message.content) {
          const content = data.message.content;
          // Handle string content
          if (typeof content === 'string') {
            onMessage(content);
          }
          // Handle ContentBlock array (TextBlock, ThinkingBlock, etc.)
          else if (Array.isArray(content)) {
            // Extract text from TextBlock items
            const text = content
              .filter((block: any) => 'text' in block)
              .map((block: any) => block.text)
              .join('');
            if (text) onMessage(text);
          }
        }
      } catch (err) {
        console.error('Failed to parse message event:', err);
      }
    });

    // Listen for 'stream_event' events (streaming updates)
    eventSource.addEventListener('stream_event', (event) => {
      try {
        const data = JSON.parse(event.data);
        // Backend sends: {"event": {...}}
        if (data.event) {
          // Handle different stream event types
          if (data.event.type === 'content_delta' && data.event.delta?.text) {
            onMessage(data.event.delta.text);
          } else if (data.event.type === 'status') {
            onStatus?.(data.event.status);
          }
        }
      } catch (err) {
        console.error('Failed to parse stream_event:', err);
      }
    });

    // Listen for 'done' event (completion)
    eventSource.addEventListener('done', () => {
      onDone();
      eventSource.close();
    });

    // Listen for 'interrupted' event (cancelled by user)
    eventSource.addEventListener('interrupted', () => {
      onInterrupted?.();
      eventSource.close();
    });

    // Listen for 'error' event
    eventSource.addEventListener('error', (event) => {
      // Check if it's a network error or an error event with data
      if (event.type === 'error' && (event as MessageEvent).data) {
        try {
          const data = JSON.parse((event as MessageEvent).data);
          onError(data.error || 'Unknown error');
        } catch {
          onError('Connection error');
        }
      } else if (eventSource.readyState === EventSource.CLOSED) {
        // Connection closed - this can happen after cancel, don't report as error
        return;
      } else {
        onError('Connection error');
      }
      eventSource.close();
    });

    return { eventSource };
  } catch (error) {
    onError(error instanceof Error ? error.message : 'Unknown error');
    return { eventSource: null };
  }
};
