/**
 * Legacy streaming transport: POST /message to stash the turn, then a browser EventSource
 * (GET /sessions/{id}/stream) to receive the SSE reply. Used by local/single-host providers
 * (Ollama, Claude Code, Codex) whose conversation state is persisted server-side.
 */

import type { MessageRequest } from '../types';
import {
  API_BASE,
  NOOP_STREAM_RESULT,
  processContentBlocks,
  type SendMessageStreamCallbacks,
  type SendMessageStreamResult,
} from './shared';
import { getDefaultHeaders } from '@mlflow/mlflow/src/common/utils/FetchUtils';

/**
 * Create an EventSource for streaming responses.
 */
export const createEventSource = (sessionId: string): EventSource => {
  return new EventSource(`${API_BASE}/sessions/${sessionId}/stream`);
};

/**
 * Attach the SSE listeners for one streaming turn. Shared by the initial send and by
 * resumeStream so a resumed turn behaves identically.
 */
const attachStreamListeners = (
  eventSource: EventSource,
  sessionId: string,
  callbacks: SendMessageStreamCallbacks,
): void => {
  const { onMessage, onError, onDone, onStatus, onToolUse, onInterrupted, onPermissionRequest } = callbacks;

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
        } else if (Array.isArray(content)) {
          // Handle ContentBlock array (TextBlock, ThinkingBlock, etc.)
          processContentBlocks(content, onMessage, onToolUse);
        }
      }
    } catch (err) {
      // fail silently
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
      // fail silently
    }
  });

  // A 'permission_request' ENDS the turn: the backend pauses by closing its side after
  // emitting the prompt. We surface it and close the stream; the user's decision is delivered
  // via resumeStream, which opens a fresh stream to continue. Stateless across the pause.
  eventSource.addEventListener('permission_request', (event) => {
    try {
      const data = JSON.parse((event as MessageEvent).data);
      onPermissionRequest?.({
        // Bind the request to the session that produced it so the decision always targets
        // the originating session.
        sessionId,
        requestId: data.request_id,
        toolName: data.tool_name,
        toolInput: data.tool_input ?? {},
      });
    } catch (err) {
      onError('Failed to read a tool permission request from the assistant.');
    }
    eventSource.close();
  });

  // Listen for 'done' event (completion). The DONE session_id for these providers is an
  // opaque server-side session token (persisted by the server), not a client-carried
  // history blob, so it is not surfaced here.
  eventSource.addEventListener('done', () => {
    onToolUse?.([]);
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
};

/**
 * Send a message and get the response stream via SSE.
 * First POSTs to /message to initiate, then connects to SSE endpoint.
 * Returns a handle whose cancel() closes the EventSource.
 */
export const sendMessageStream = async (
  request: MessageRequest,
  callbacks: SendMessageStreamCallbacks,
): Promise<SendMessageStreamResult> => {
  const { onError, onSessionId } = callbacks;

  try {
    // Step 1: POST the message to initiate processing
    // eslint-disable-next-line no-restricted-globals -- See go/spog-fetch
    const response = await fetch(`${API_BASE}/message`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getDefaultHeaders(document.cookie),
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.text();
      onError(`Failed to send message: ${error}`);
      return NOOP_STREAM_RESULT;
    }

    // Step 2: Get the session_id from the response
    const result = await response.json();
    const sessionId = result.session_id;

    if (!sessionId) {
      onError('No session_id returned from server');
      return NOOP_STREAM_RESULT;
    }

    // Notify caller of the session ID
    onSessionId?.(sessionId);

    // Step 3: Connect to the SSE endpoint to receive the stream
    const eventSource = createEventSource(sessionId);
    attachStreamListeners(eventSource, sessionId, callbacks);
    return { cancel: () => eventSource.close() };
  } catch (error) {
    onError(error instanceof Error ? error.message : 'Unknown error');
    return NOOP_STREAM_RESULT;
  }
};

/**
 * Resume a turn paused at a permission prompt: POST the decision, then open a fresh stream
 * that continues from where the turn left off. Stateless across the pause — any server can
 * serve the resume because the pending decision lives on the session, not in process memory.
 */
export const resumeStream = async (
  sessionId: string,
  requestId: string,
  decision: 'allow' | 'deny',
  callbacks: SendMessageStreamCallbacks,
): Promise<SendMessageStreamResult> => {
  try {
    // eslint-disable-next-line no-restricted-globals -- See go/spog-fetch
    const response = await fetch(`${API_BASE}/sessions/${sessionId}/permission`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getDefaultHeaders(document.cookie),
      },
      body: JSON.stringify({ request_id: requestId, decision }),
    });
    if (!response.ok) {
      callbacks.onError('Failed to send your permission decision. Please try again.');
      return NOOP_STREAM_RESULT;
    }
  } catch (error) {
    callbacks.onError('Failed to send your permission decision. Please try again.');
    return NOOP_STREAM_RESULT;
  }

  const eventSource = createEventSource(sessionId);
  attachStreamListeners(eventSource, sessionId, callbacks);
  return { cancel: () => eventSource.close() };
};
