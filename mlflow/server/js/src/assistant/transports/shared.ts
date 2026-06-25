/**
 * Shared primitives for the Assistant streaming transports.
 *
 * This is a leaf module: both transports (eventSourceTransport, fetchStreamTransport) import
 * from here, and neither imports the other or the AssistantService REST client.
 */

import type { PermissionRequest, ToolResultInfo, ToolUseInfo } from '../types';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';

export const API_BASE = getAjaxUrl('ajax-api/3.0/mlflow/assistant');

export interface SendMessageStreamCallbacks {
  onMessage: (text: string) => void;
  /** `code` is the backend's machine-readable error class when it provided one. */
  onError: (error: string, code?: string) => void;
  onDone: () => void;
  onStatus?: (status: string) => void;
  onSessionId?: (sessionId: string) => void;
  onToolUse?: (tools: ToolUseInfo[]) => void;
  onToolResult?: (result: ToolResultInfo) => void;
  onInterrupted?: () => void;
  /** Called with the updated conversation history blob from the DONE event (OpenAI-compatible). */
  onConversationHistory?: (history: string) => void;
  /** Called when the assistant pauses at a tool-call permission prompt awaiting an allow/deny. */
  onPermissionRequest?: (request: PermissionRequest) => void;
  onUsage?: (usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    cache_read_tokens?: number;
    total_cost_usd?: number | null;
  }) => void;
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

/** Tool-result `content` is string | list[dict] | null on the wire; collapse to a string. */
const normalizeToolResultContent = (content: unknown): string => {
  if (content == null) return '';
  if (typeof content === 'string') return content;
  return JSON.stringify(content, null, 2);
};

/**
 * Process a content block array from an assistant response, emitting text, tool-use,
 * and tool-result blocks in order so the transcript preserves their sequence.
 */
export const processContentBlocks = (
  content: any[],
  onMessage: (text: string) => void,
  onToolUse?: (tools: ToolUseInfo[]) => void,
  onToolResult?: (result: ToolResultInfo) => void,
): void => {
  for (const block of content) {
    if ('text' in block && block.text) {
      onMessage(block.text);
    } else if (block.tool_use_id) {
      // ToolResultBlock: carries the output for a previously-streamed tool call.
      onToolResult?.({
        toolUseId: block.tool_use_id,
        content: normalizeToolResultContent(block.content),
        isError: Boolean(block.is_error),
      });
    } else if (block.name && block.input) {
      // TextBlock-less ToolUseBlock (claude_code & openai_compatible both shape it this way).
      onToolUse?.([
        {
          id: block.id,
          name: block.name,
          description: block.input?.description,
          input: block.input,
        },
      ]);
    }
  }
};
