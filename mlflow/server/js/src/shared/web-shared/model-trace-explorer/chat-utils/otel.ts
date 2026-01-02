import { get, has, isArray, isNil, isObject, isString } from 'lodash';

import type { ModelTraceChatMessage, ModelTraceContentParts, ModelTraceToolCall } from '../ModelTrace.types';
import { prettyPrintChatMessage } from '../ModelTraceExplorer.utils';

// This file normalizes OpenTelemetry GenAI semantic convention input messages
// Schema reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-input-messages.json
// Supported parts: TextPart, ToolCallRequestPart, ToolCallResponsePart
// Not supported (rejected): BlobPart, FilePart, UriPart, ReasoningPart, GenericPart

type OtelBasePart = { type: string } & Record<string, unknown>;

type OtelTextPart = OtelBasePart & { type: 'text'; content: string };

type OtelToolCallRequestPart = OtelBasePart & {
  type: 'tool_call' | 'function_call';
  id?: string | null;
  name: string;
  arguments?: unknown;
};

type OtelToolCallResponsePart = OtelBasePart & {
  type: 'tool_call_response';
  id?: string | null;
  response: unknown;
};

type OTelGenAIPart = OtelTextPart | OtelToolCallRequestPart | OtelToolCallResponsePart;

type OtelGenAIMessage = {
  role: 'system' | 'user' | 'assistant' | 'tool';
  name?: string;
  parts: OTelGenAIPart[];
};

const isOtelTextPart = (obj: unknown): obj is OtelTextPart => {
  return isObject(obj) && get(obj, 'type') === 'text' && isString(get(obj, 'content'));
};

const isOtelToolCallRequestPart = (obj: unknown): obj is OtelToolCallRequestPart => {
  const type = get(obj, 'type');
  if (!isObject(obj) || !(type === 'tool_call' || type === 'function_call')) return false;
  if (!isString(get(obj, 'name'))) return false;
  const id = get(obj, 'id');
  return isNil(id) || isString(id);
};

const isOtelToolCallResponsePart = (obj: unknown): obj is OtelToolCallResponsePart => {
  if (!isObject(obj) || get(obj, 'type') !== 'tool_call_response') return false;
  const id = get(obj, 'id');
  if (!(isNil(id) || isString(id))) return false;
  return has(obj, 'response');
};

const isSupportedOtelPart = (obj: unknown): obj is OTelGenAIPart => {
  return isOtelTextPart(obj) || isOtelToolCallRequestPart(obj) || isOtelToolCallResponsePart(obj);
};

export const isOtelGenAIChatMessage = (obj: unknown): obj is OtelGenAIMessage => {
  if (!isObject(obj)) return false;
  const role = get(obj, 'role');
  if (!isString(role) || !['system', 'user', 'assistant', 'tool'].includes(role)) return false;
  if (!has(obj, 'parts') || !isArray((obj as any).parts) || (obj as any).parts.length === 0) return false;
  return (obj as any).parts.every(isSupportedOtelPart);
};

const normalizeToolCallRequestPart = (part: OtelToolCallRequestPart): ModelTraceToolCall => {
  const args = get(part, 'arguments') as unknown;
  let argumentsStr = '';
  try {
    argumentsStr = JSON.stringify(args ?? {});
  } catch {
    argumentsStr = String(args);
  }
  return {
    id: String(get(part, 'id') ?? get(part, 'name') ?? ''),
    function: {
      name: String(get(part, 'name')),
      arguments: argumentsStr,
    },
  };
};
const normalizeToolCallResponsePart = (part: OtelToolCallResponsePart): ModelTraceChatMessage => {
  const callId = String(get(part, 'id') ?? '');
  const response = get(part, 'response');
  const content = isString(response)
    ? response
    : (() => {
        try {
          return JSON.stringify(response);
        } catch {
          return String(response);
        }
      })();
  return {
    role: 'tool',
    tool_call_id: callId,
    content,
  };
};

// Convert a single OTEL GenAI message into a single UI message.
export const normalizeOtelGenAIChatMessage = (obj: OtelGenAIMessage): ModelTraceChatMessage | null => {
  if (!isOtelGenAIChatMessage(obj)) return null;
  const role: ModelTraceChatMessage['role'] = obj.role as any;

  const contentParts: ModelTraceContentParts[] = [];
  const toolCalls: ModelTraceToolCall[] = [];
  let toolResultMessage: ModelTraceChatMessage | null = null;

  for (const part of obj.parts) {
    if (isOtelTextPart(part)) {
      contentParts.push({ type: 'text', text: part.content });
      continue;
    }
    if (isOtelToolCallRequestPart(part)) {
      toolCalls.push(normalizeToolCallRequestPart(part));
      continue;
    }
    if (isOtelToolCallResponsePart(part)) {
      toolResultMessage = normalizeToolCallResponsePart(part);
      continue;
    }
  }

  if (toolResultMessage && contentParts.length === 0 && toolCalls.length === 0) {
    return toolResultMessage;
  }

  return prettyPrintChatMessage({
    type: 'message',
    role,
    ...(obj.name && { name: obj.name }),
    ...(contentParts.length > 0 && { content: contentParts }),
    ...(toolCalls.length > 0 && { tool_calls: toolCalls }),
  });
};
