/**
 * Transcript parsing utilities for Qwen Code JSONL files.
 *
 * Qwen Code writes chat records to
 * `~/.qwen/projects/<project-id>/chats/<sessionId>.jsonl`. Records have
 * `uuid`/`parentUuid` for tree traversal but are also emitted in
 * chronological order, so we walk them sequentially for span creation and
 * use the tree only for message history reconstruction.
 */

import { readFileSync } from 'node:fs';
import type {
  ChatRecord,
  FunctionCall,
  FunctionCallPart,
  FunctionResponsePart,
  GeminiMessage,
  GeminiPart,
  TextPart,
  UsageMetadata,
} from './types.js';

export const NANOSECONDS_PER_MS = 1e6;

/** Read and parse a Qwen Code JSONL transcript file. */
export function readTranscript(path: string): ChatRecord[] {
  const content = readFileSync(path, 'utf-8');
  return content
    .split('\n')
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line) as ChatRecord);
}

/** Parse an ISO timestamp string to nanoseconds since Unix epoch. */
export function parseTimestampToNs(timestamp: string | undefined | null): number | null {
  if (!timestamp) {
    return null;
  }
  try {
    const ms = new Date(timestamp).getTime();
    if (isNaN(ms)) {
      return null;
    }
    return ms * NANOSECONDS_PER_MS;
  } catch {
    return null;
  }
}

/**
 * Narrow a GeminiPart union member. The runtime shape is discriminated by
 * the presence of `text` / `functionCall` / `functionResponse` keys.
 */
export function isTextPart(part: GeminiPart): part is TextPart {
  return typeof (part as TextPart).text === 'string';
}

export function isFunctionCallPart(part: GeminiPart): part is FunctionCallPart {
  return (part as FunctionCallPart).functionCall != null;
}

export function isFunctionResponsePart(part: GeminiPart): part is FunctionResponsePart {
  return (part as FunctionResponsePart).functionResponse != null;
}

/**
 * Extract user-facing text from a record's message. Internal reasoning
 * (`thought: true` text parts) is excluded — those should not appear as
 * assistant content in the Chat view.
 *
 * If `includeThoughts` is true, thought parts are included (used only by
 * request-preview fallbacks where there's nothing else to show).
 */
export function getMessageText(record: ChatRecord, includeThoughts = false): string {
  const msg = record.message;
  if (typeof msg === 'string') {
    return msg;
  }
  if (!isGeminiMessage(msg)) {
    return '';
  }
  return msg.parts
    .filter(isTextPart)
    .filter((p) => includeThoughts || !p.thought)
    .map((p) => p.text)
    .join('\n');
}

/** Return the functionCall parts embedded in an assistant record's message. */
export function getFunctionCalls(record: ChatRecord): FunctionCall[] {
  const msg = record.message;
  if (!isGeminiMessage(msg)) {
    return [];
  }
  return msg.parts.filter(isFunctionCallPart).map((p) => p.functionCall);
}

function isGeminiMessage(msg: unknown): msg is GeminiMessage {
  return (
    typeof msg === 'object' &&
    msg != null &&
    'parts' in msg &&
    Array.isArray((msg as GeminiMessage).parts)
  );
}

/**
 * Return the records belonging to the last turn (from the last user record
 * through end-of-file), in chronological order.
 */
export function getLastTurnRecords(records: ChatRecord[]): ChatRecord[] {
  for (let i = records.length - 1; i >= 0; i--) {
    if (records[i].type === 'user' && getMessageText(records[i]).trim()) {
      return records.slice(i);
    }
  }
  return [];
}

/** Build a lookup from tool call_id to its `tool_result` record. */
export function buildToolResultMap(records: ChatRecord[]): Map<string, ChatRecord> {
  const byCallId = new Map<string, ChatRecord>();
  for (const record of records) {
    if (record.type === 'tool_result' && record.toolCallResult?.callId) {
      byCallId.set(record.toolCallResult.callId, record);
    }
  }
  return byCallId;
}

/** Extract structured token usage from a record's usageMetadata, if any. */
export function getTokenUsage(
  metadata: UsageMetadata | undefined,
): { input: number; output: number; total: number } | null {
  if (!metadata) {
    return null;
  }
  const input = metadata.promptTokenCount ?? metadata.input_tokens ?? 0;
  const output = metadata.candidatesTokenCount ?? metadata.output_tokens ?? 0;
  const total = metadata.totalTokenCount ?? input + output;
  if (input === 0 && output === 0) {
    return null;
  }
  return { input, output, total };
}

/**
 * Normalize a `resultDisplay` value (which can be a plain string or a
 * structured object) into a string suitable for a TOOL span output or a
 * tool message's content.
 */
export function formatResultDisplay(display: unknown): string {
  if (display == null) {
    return '';
  }
  if (typeof display === 'string') {
    return display;
  }
  return JSON.stringify(display);
}

/**
 * Return a string rendering of a tool_result record's output. Prefers
 * `toolCallResult.resultDisplay` (the user-facing rendering) and falls back
 * to the raw `functionResponse.response` payload embedded in `message.parts`
 * when `resultDisplay` is omitted. Returns an empty string if neither is
 * available.
 */
export function getToolOutput(record: ChatRecord): string {
  const display = record.toolCallResult?.resultDisplay;
  if (display != null) {
    return formatResultDisplay(display);
  }
  const msg = record.message;
  if (isGeminiMessage(msg)) {
    for (const part of msg.parts) {
      if (isFunctionResponsePart(part) && part.functionResponse.response != null) {
        return formatResultDisplay(part.functionResponse.response);
      }
    }
  }
  return '';
}
