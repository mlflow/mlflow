/**
 * Shared primitives for the Assistant streaming transports.
 *
 * This is a leaf module: both transports (eventSourceTransport, fetchStreamTransport) import
 * from here, and neither imports the other or the AssistantService REST client.
 */

import type { PermissionRequest, ToolUseInfo } from '../types';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';

export const API_BASE = getAjaxUrl('ajax-api/3.0/mlflow/assistant');

export interface SendMessageStreamCallbacks {
  onMessage: (text: string) => void;
  onError: (error: string) => void;
  onDone: () => void;
  onStatus?: (status: string) => void;
  onSessionId?: (sessionId: string) => void;
  onToolUse?: (tools: ToolUseInfo[]) => void;
  onInterrupted?: () => void;
  /** Called with the updated conversation history blob from the DONE event (OpenAI-compatible). */
  onConversationHistory?: (history: string) => void;
  /** Called when the assistant pauses at a tool-call permission prompt awaiting an allow/deny. */
  onPermissionRequest?: (request: PermissionRequest) => void;
}

export interface SendMessageStreamResult {
  /** Cancel the in-flight stream (closes the EventSource or aborts the fetch). */
  cancel: () => void;
}

export const NOOP_STREAM_RESULT: SendMessageStreamResult = { cancel: () => {} };

/**
 * Parse a single SSE frame ("event: <type>\ndata: <json>") into its event name and JSON data.
 * Returns null when the frame carries no parseable data line (e.g. comments or heartbeats).
 */
const parseSseFrame = (frame: string): { event: string; data: any } | null => {
  let event = 'message';
  const dataLines: string[] = [];
  for (const line of frame.split(/\r?\n/)) {
    if (line.startsWith('event:')) {
      event = line.slice('event:'.length).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice('data:'.length).trim());
    }
  }
  if (dataLines.length === 0) {
    return null;
  }
  try {
    return { event, data: JSON.parse(dataLines.join('\n')) };
  } catch {
    return null;
  }
};

/**
 * Read an SSE byte stream (from a fetch ReadableStream) as a sequence of parsed frames.
 *
 * Owns the reader loop: decodes UTF-8 (buffering multibyte sequences split across chunks via
 * the streaming TextDecoder), splits on blank-line frame boundaries while holding back a trailing
 * partial frame for the next chunk, and yields each parseable frame. Tolerates LF and CRLF line
 * endings. This is framing only — it has no knowledge of any particular event type. Aborting the
 * underlying request rejects the in-flight read, which propagates out of this generator for the
 * caller to handle.
 */
export async function* readSseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<{ event: string; data: any }> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) {
      // Flush the decoder and try to parse any leftover buffer: a stream that ends without a
      // trailing blank line (e.g. a proxy truncated right after the data line) would otherwise
      // drop a final terminal `done`/`error` frame. parseSseFrame returns null for an empty or
      // incomplete buffer, so this is a no-op in the normal case.
      buffer += decoder.decode();
      const parsed = parseSseFrame(buffer);
      if (parsed) {
        yield parsed;
      }
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop() ?? '';
    for (const frame of frames) {
      const parsed = parseSseFrame(frame);
      if (parsed) {
        yield parsed;
      }
    }
  }
}

/**
 * Process content block array from assistant response.
 * Extracts text or tool uses and calls appropriate callbacks.
 */
export const processContentBlocks = (
  content: any[],
  onMessage: (text: string) => void,
  onToolUse?: (tools: ToolUseInfo[]) => void,
): void => {
  // Extract text from TextBlock items
  const text = content
    .filter((block: any) => 'text' in block)
    .map((block: any) => block.text)
    .join('');

  if (text) {
    // Clear tools and show text when assistant is responding
    onToolUse?.([]);
    onMessage(text);
    return;
  }

  // Only show tool uses when there's no text response yet
  const toolUses = content
    .filter((block: any) => block.name && block.input && !block.tool_use_id)
    .map((block: any) => ({
      id: block.id,
      name: block.name,
      description: block.input?.description,
      input: block.input,
    }));
  if (toolUses.length > 0 && onToolUse) {
    onToolUse(toolUses);
  }
};
