import { get, has, isArray, isNil, isObject, isString } from 'lodash';

import type { ModelTraceChatMessage, ModelTraceContentParts, ModelTraceToolCall } from '../ModelTrace.types';
import { prettyPrintChatMessage } from '../ModelTraceExplorer.utils';

// This file normalizes OpenTelemetry GenAI semantic convention input messages
// Schema reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-input-messages.json
// Parts without a dedicated renderer are preserved as JSON in the chat content.

type OtelBasePart = { type: string } & Record<string, unknown>;

type OtelTextPart = OtelBasePart & { type: 'text'; content: string };

type OtelToolCallRequestPart = OtelBasePart & {
  type: 'tool_call';
  id?: string | null;
  name: string;
  arguments?: unknown;
};

type OtelToolCallResponsePart = OtelBasePart & {
  type: 'tool_call_response';
  id?: string | null;
  response: unknown;
};

type OtelGenAIMessage = {
  role: string;
  name?: string;
  parts: OtelBasePart[];
};

const isOtelTextPart = (obj: unknown): obj is OtelTextPart => {
  return isObject(obj) && get(obj, 'type') === 'text' && isString(get(obj, 'content'));
};

const isOtelToolCallRequestPart = (obj: unknown): obj is OtelToolCallRequestPart => {
  if (!isObject(obj) || get(obj, 'type') !== 'tool_call') return false;
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

const isOtelPart = (obj: unknown): obj is OtelBasePart => {
  return isObject(obj) && !isArray(obj) && isString(get(obj, 'type'));
};

export const isOtelGenAIChatMessage = (obj: unknown): obj is OtelGenAIMessage => {
  if (!isObject(obj)) return false;
  const role = get(obj, 'role');
  if (!isString(role)) return false;
  const parts = get(obj, 'parts') as unknown;
  return isArray(parts) && parts.every(isOtelPart);
};

const serializeValue = (value: unknown): string => {
  try {
    return JSON.stringify(value, null, 2) ?? String(value);
  } catch {
    return String(value);
  }
};

const normalizeGenericPart = (part: OtelBasePart): ModelTraceContentParts => {
  return { type: 'text', text: `\`\`\`json\n${serializeValue(part)}\n\`\`\`` };
};

const escapeMarkdownUrl = (url: string): string => {
  return url.replace(/[\s()<>\\]/g, (character) => {
    if (character === '(') return '%28';
    if (character === ')') return '%29';
    return encodeURIComponent(character);
  });
};

const normalizeMediaPart = (part: OtelBasePart): ModelTraceContentParts | null => {
  const uri = part['uri'];
  const modality = part['modality'];
  if (part.type === 'uri' && isString(uri)) {
    try {
      const parsedUrl = new URL(uri);
      const protocol = parsedUrl.protocol;
      if (protocol === 'mlflow-attachment:' || (modality === 'image' && ['http:', 'https:'].includes(protocol))) {
        // Escape markdown destination delimiters while preserving existing URL encoding.
        const url = escapeMarkdownUrl(parsedUrl.href);
        return { type: 'image_url', image_url: { url } };
      }
    } catch {
      // Invalid or provider-specific URIs remain inspectable in the JSON fallback.
    }
  }

  const rawContent = part['content'];
  if (part.type !== 'blob' || !isString(rawContent)) return null;
  if (rawContent.startsWith('mlflow-attachment://')) {
    return { type: 'image_url', image_url: { url: escapeMarkdownUrl(rawContent) } };
  }

  const content = rawContent.replace(/\s/g, '');
  if (!content || !/^[A-Za-z0-9+/]*={0,2}$/.test(content)) return null;
  const rawMimeType = part['mime_type'];
  const mimeType = isString(rawMimeType) ? rawMimeType.toLowerCase() : '';
  if (modality === 'image' && /^image\/(png|jpeg|gif|webp|avif|bmp)$/.test(mimeType)) {
    return { type: 'image_url', image_url: { url: `data:${mimeType};base64,${content}` } };
  }
  if (modality === 'audio') {
    const format = ['audio/wav', 'audio/x-wav'].includes(mimeType)
      ? 'wav'
      : ['audio/mpeg', 'audio/mp3'].includes(mimeType)
        ? 'mp3'
        : null;
    if (format) return { type: 'input_audio', input_audio: { data: content, format } };
  }
  return null;
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
  const role: ModelTraceChatMessage['role'] = obj.role.toLowerCase() as any;

  const contentParts: ModelTraceContentParts[] = [];
  const toolCalls: ModelTraceToolCall[] = [];
  const reasoningParts: string[] = [];

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
      if (obj.parts.length === 1) return normalizeToolCallResponsePart(part);
      contentParts.push(normalizeGenericPart(part));
      continue;
    }
    const content = part['content'];
    if (part.type === 'reasoning' && isString(content)) {
      reasoningParts.push(content);
      continue;
    }
    contentParts.push(normalizeMediaPart(part) ?? normalizeGenericPart(part));
  }

  return prettyPrintChatMessage({
    type: 'message',
    role,
    ...(obj.name && { name: obj.name }),
    ...(contentParts.length > 0 && { content: contentParts }),
    ...(toolCalls.length > 0 && { tool_calls: toolCalls }),
    ...(reasoningParts.length > 0 && { reasoning: reasoningParts.join('\n\n') }),
  });
};
