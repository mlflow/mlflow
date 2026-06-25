/**
 * Stateless streaming transport: a single streaming POST to /chat read via fetch +
 * ReadableStream. Used by client-carried-history providers (e.g. MLflow AI Gateway) — the
 * full conversation history travels in the request body and the SSE reply streams back on the
 * same response, so the server holds no per-session state.
 */

import type { ChatRequest } from '../types';
import {
  API_BASE,
  processContentBlocks,
  readSseFrames,
  type SendMessageStreamCallbacks,
  type SendMessageStreamResult,
} from './shared';
import { getDefaultHeaders } from '@mlflow/mlflow/src/common/utils/FetchUtils';

/**
 * Per-turn state threaded through dispatchSseFrame. The provider emits a `permission_request`
 * followed by a `done` carrying the paused history, so the done handler needs to know a pause
 * happened this turn to avoid finalizing it.
 */
interface TurnState {
  sawPermissionRequest: boolean;
}

/**
 * Dispatch one SSE frame to the matching callback. Returns true when the frame is a *terminal*
 * event — one that ends this stream (done/error/interrupted) and after which the server sends
 * nothing more. This is the single source of truth for "what is terminal": callers read the
 * return value instead of re-listing event names, so the two can't drift.
 */
const dispatchSseFrame = (
  event: string,
  data: any,
  callbacks: SendMessageStreamCallbacks,
  state: TurnState,
): boolean => {
  const { onMessage, onError, onDone, onStatus, onToolUse, onInterrupted, onConversationHistory, onPermissionRequest } =
    callbacks;
  switch (event) {
    case 'message': {
      const content = data.message?.content;
      if (typeof content === 'string') {
        onMessage(content);
      } else if (Array.isArray(content)) {
        processContentBlocks(content, onMessage, onToolUse);
      }
      return false;
    }
    case 'stream_event': {
      if (data.event?.type === 'content_delta' && data.event.delta?.text) {
        onMessage(data.event.delta.text);
      } else if (data.event?.type === 'status') {
        onStatus?.(data.event.status);
      }
      return false;
    }
    case 'permission_request': {
      onPermissionRequest?.({
        // No sessionId: the stateless path replays the decision with the client-carried history,
        // not against a server session.
        requestId: data.request_id,
        toolName: data.tool_name,
        toolInput: data.tool_input ?? {},
      });
      state.sawPermissionRequest = true;
      // Not terminal: the provider still emits a DONE carrying the paused history (with the
      // unresolved tool_call) so a later resume can continue from it.
      return false;
    }
    case 'done': {
      // For client-carried-history providers the DONE session_id is the updated history blob.
      if (data.session_id) {
        onConversationHistory?.(data.session_id);
      }
      // A DONE that follows a permission_request is a *pause*, not a completion: skip onDone so
      // the Allow/Deny prompt stays up and isStreaming stays true. The decision is replayed via a
      // fresh /chat POST carrying the history + tool_decisions. The stream still ends here (the
      // socket closes), so this is terminal for the read loop.
      if (state.sawPermissionRequest) {
        return true;
      }
      onToolUse?.([]);
      onDone();
      return true;
    }
    case 'interrupted':
      onInterrupted?.();
      return true;
    case 'error':
      onError(data.error || 'Unknown error');
      return true;
    default:
      return false;
  }
};

// If no byte arrives for this long, treat the stream as dead and surface an error. Generous on
// purpose: an LLM turn can legitimately go quiet (slow first token, a tool running), so this only
// catches a connection that has truly stalled — not one that's merely slow.
const INACTIVITY_MS = 60_000;

/**
 * Stream a chat turn over a single POST using fetch + ReadableStream.
 *
 * The read loop runs in the background; the returned handle's cancel() aborts the in-flight
 * request.
 */
export const streamChatViaFetch = async (
  request: ChatRequest,
  callbacks: SendMessageStreamCallbacks,
): Promise<SendMessageStreamResult> => {
  const controller = new AbortController();
  const cancel = () => controller.abort();

  let response: Response;
  try {
    // eslint-disable-next-line no-restricted-globals -- See go/spog-fetch
    response = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getDefaultHeaders(document.cookie),
      },
      body: JSON.stringify(request),
      signal: controller.signal,
    });
  } catch (error) {
    if (!controller.signal.aborted) {
      callbacks.onError(error instanceof Error ? error.message : 'Unknown error');
    }
    return { cancel };
  }

  if (!response.ok || !response.body) {
    const error = await response.text().catch(() => response.statusText);
    callbacks.onError(`Failed to send message: ${error}`);
    return { cancel };
  }

  // Read in the background so the caller gets the cancel handle immediately.
  const body = response.body;
  void (async () => {
    let watchdog: ReturnType<typeof setTimeout> | undefined;
    const clearWatchdog = () => {
      if (watchdog) clearTimeout(watchdog);
      watchdog = undefined;
    };
    // Aborting the request rejects the in-flight read; the catch below sees signal.aborted and
    // stays quiet, so the error we raise here is the one the UI shows.
    const armWatchdog = () =>
      (watchdog = setTimeout(() => {
        callbacks.onError('The response stalled. Please try again.');
        controller.abort();
      }, INACTIVITY_MS));

    let sawTerminal = false;
    const turnState: TurnState = { sawPermissionRequest: false };
    try {
      armWatchdog();
      for await (const { event, data } of readSseFrames(body)) {
        if (dispatchSseFrame(event, data, callbacks, turnState)) {
          sawTerminal = true;
          clearWatchdog(); // turn is done; a slow socket close shouldn't trip the watchdog
        } else {
          armWatchdog(); // a frame arrived — the stream is alive, reset the inactivity clock
        }
      }
      // The body reached EOF. If the server never sent a terminal event, the turn didn't
      // complete (provider crash, dropped proxy, truncated stream) — surface it so the UI
      // finalizes instead of spinning forever.
      if (!sawTerminal && !controller.signal.aborted) {
        callbacks.onError('The response ended unexpectedly. Please try again.');
      }
    } catch (error) {
      // An abort (user cancel / reset / watchdog) is expected — only surface real failures.
      if (!controller.signal.aborted) {
        callbacks.onError(error instanceof Error ? error.message : 'Connection error');
      }
    } finally {
      clearWatchdog();
    }
  })();

  return { cancel };
};
