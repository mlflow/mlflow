import { compact, has, isArray, isObject, isString } from 'lodash';

import type { ModelTraceChatMessage, ModelTraceContentParts, ModelTraceToolCall } from '../ModelTrace.types';
import { prettyPrintChatMessage } from '../ModelTraceExplorer.utils';

export type GeminiChatInput = {
  contents: string | GeminiContent[];
};

export type GeminiChatOutput = {
  candidates: GeminiCandidate[];
  // propmtFeedback: GeminiPropmptFeedback;
  // usageMetadata: GeminiUsageMetadata;
  modelVersion: string;
  responseId: string;
};

type GeminiCandidate = {
  content: GeminiContent;
  finishReason:
    | 'FINISH_REASON_UNSPECIFIED'
    | 'STOP'
    | 'MAX_TOKENS'
    | 'SAFETY'
    | 'RECITATION'
    | 'LANGUAGE'
    | 'OTHER'
    | 'BLOCKLIST'
    | 'PROHIBITED_CONTENT'
    | 'SPII'
    | 'MALFORMED_FUNCTION_CAL'
    | 'IMAGE_SAFETY'
    | 'UNEXPECTED_TOOL_CAL';
  // safetyRatings: GeminiSafetyRating[]
  // citationMetadata: GeminiCitationMetadata
  // tokenCount: number
  // groundingAttributions: GeminiGroundingAttribution[]
  // groundingMetadata: GeminiGroundingMetadata
  // avgLogprobs: number
  // logprobsResult: GeminiLogprobsResult
  // urlContextMetadata: GeminiUrlContextMetadata
  // index: number
};

type GeminiContent = {
  role: 'user' | 'model';
  parts: GeminiContentPart[];
};

type GeminiTextPart = {
  text: string;
  thought?: boolean | null;
};

type GeminiFunctionCallPart = {
  function_call: {
    name: string;
    args: Record<string, unknown>;
  };
};

type GeminiFunctionResponsePart = {
  function_response: {
    name: string;
    response: Record<string, unknown>;
  };
};

type GeminiInlineDataPart = {
  inline_data: {
    data: string;
    mime_type: string;
  };
};

type GeminiFileDataPart = {
  file_data: {
    file_uri: string;
    mime_type: string;
  };
};

type GeminiContentPart =
  | GeminiTextPart
  | GeminiFunctionCallPart
  | GeminiFunctionResponsePart
  | GeminiInlineDataPart
  | GeminiFileDataPart;

const isGeminiTextPart = (obj: unknown): obj is GeminiTextPart => {
  return isObject(obj) && 'text' in obj && isString(obj.text);
};

const isGeminiFunctionCallPart = (obj: unknown): obj is GeminiFunctionCallPart => {
  return isObject(obj) && 'function_call' in obj && isObject((obj as any).function_call);
};

const isGeminiFunctionResponsePart = (obj: unknown): obj is GeminiFunctionResponsePart => {
  return isObject(obj) && 'function_response' in obj && isObject((obj as any).function_response);
};

const isGeminiInlineDataPart = (obj: unknown): obj is GeminiInlineDataPart => {
  return (
    isObject(obj) &&
    'inline_data' in obj &&
    isObject((obj as any).inline_data) &&
    isString((obj as any).inline_data.data) &&
    isString((obj as any).inline_data.mime_type)
  );
};

const isGeminiFileDataPart = (obj: unknown): obj is GeminiFileDataPart => {
  return (
    isObject(obj) &&
    'file_data' in obj &&
    isObject((obj as any).file_data) &&
    isString((obj as any).file_data.file_uri) &&
    isString((obj as any).file_data.mime_type)
  );
};

const isGeminiContentPart = (obj: unknown): obj is GeminiContentPart => {
  return (
    isGeminiTextPart(obj) ||
    isGeminiFunctionCallPart(obj) ||
    isGeminiFunctionResponsePart(obj) ||
    isGeminiInlineDataPart(obj) ||
    isGeminiFileDataPart(obj)
  );
};

const isThinkingPart = (part: GeminiTextPart): boolean => {
  return part.thought === true;
};

const isGeminiContent = (obj: unknown): obj is GeminiContent => {
  return (
    isObject(obj) &&
    'role' in obj &&
    isString(obj.role) &&
    ['user', 'model'].includes(obj.role) &&
    has(obj, 'parts') &&
    Array.isArray(obj.parts) &&
    obj.parts.every(isGeminiContentPart)
  );
};

// Gemini SDK serializes bytes as Python literal "b'...'" with escaped hex bytes.
// This function detects that format, decodes the Python bytes, and returns base64.
// If the data is already base64, it is returned as-is.
const cleanBase64Data = (data: string): string => {
  const match = data.match(/^b'(.*)'$/s);
  if (!match) return data;

  const inner = match[1];
  // Check if this contains Python hex escapes (e.g., \x89, \r, \n)
  if (/\\x[0-9a-fA-F]{2}|\\[rnt\\']/.test(inner)) {
    return decodePythonBytesLiteral(inner);
  }
  // Otherwise it's already base64 wrapped in b'...'
  return inner;
};

const decodePythonBytesLiteral = (inner: string): string => {
  const bytes: number[] = [];
  for (let i = 0; i < inner.length; i++) {
    if (inner[i] === '\\' && i + 1 < inner.length) {
      const next = inner[i + 1];
      if (next === 'x' && i + 3 < inner.length) {
        bytes.push(parseInt(inner.substring(i + 2, i + 4), 16));
        i += 3;
      } else if (next === 'n') {
        bytes.push(0x0a);
        i += 1;
      } else if (next === 'r') {
        bytes.push(0x0d);
        i += 1;
      } else if (next === 't') {
        bytes.push(0x09);
        i += 1;
      } else if (next === '\\') {
        bytes.push(0x5c);
        i += 1;
      } else if (next === "'") {
        bytes.push(0x27);
        i += 1;
      } else {
        bytes.push(inner.charCodeAt(i));
      }
    } else {
      bytes.push(inner.charCodeAt(i));
    }
  }

  // Convert byte array to base64
  let binary = '';
  for (const b of bytes) {
    binary += String.fromCharCode(b);
  }
  return btoa(binary);
};

const getAudioFormat = (mimeType: string): 'wav' | 'mp3' | null => {
  if (mimeType === 'audio/wav' || mimeType === 'audio/x-wav') return 'wav';
  if (mimeType === 'audio/mpeg' || mimeType === 'audio/mp3') return 'mp3';
  return null;
};

const isGeminiCandidate = (obj: unknown): obj is GeminiCandidate => {
  return isObject(obj) && 'content' in obj && isGeminiContent(obj.content);
};

const processGeminiContentParts = (
  parts: GeminiContentPart[],
): {
  textParts: ModelTraceContentParts[];
  thinking: string | null;
  toolCalls: ModelTraceToolCall[];
  functionResponses: { name: string; response: Record<string, unknown> }[];
} => {
  const textParts: ModelTraceContentParts[] = [];
  const thinkingParts: string[] = [];
  const toolCalls: ModelTraceToolCall[] = [];
  const functionResponses: { name: string; response: Record<string, unknown> }[] = [];

  for (const part of parts) {
    if (isGeminiFunctionCallPart(part)) {
      toolCalls.push({
        id: part.function_call.name,
        function: {
          name: part.function_call.name,
          arguments: JSON.stringify(part.function_call.args),
        },
      });
    } else if (isGeminiFunctionResponsePart(part)) {
      functionResponses.push({
        name: part.function_response.name,
        response: part.function_response.response,
      });
    } else if (isGeminiTextPart(part)) {
      if (isThinkingPart(part)) {
        thinkingParts.push(part.text);
      } else {
        textParts.push({ type: 'text', text: part.text });
      }
    } else if (isGeminiInlineDataPart(part)) {
      const { mime_type } = part.inline_data;
      const data = cleanBase64Data(part.inline_data.data);
      if (mime_type.startsWith('image/')) {
        textParts.push({
          type: 'image_url',
          image_url: { url: `data:${mime_type};base64,${data}` },
        });
      } else if (mime_type.startsWith('audio/')) {
        const format = getAudioFormat(mime_type);
        if (format) {
          textParts.push({
            type: 'input_audio',
            input_audio: { data, format },
          });
        }
      }
    } else if (isGeminiFileDataPart(part)) {
      const { file_uri, mime_type } = part.file_data;
      if (mime_type.startsWith('image/')) {
        textParts.push({
          type: 'image_url',
          image_url: { url: file_uri },
        });
      }
    }
  }

  const thinking = thinkingParts.length > 0 ? thinkingParts.join('\n\n') : null;
  return { textParts, thinking, toolCalls, functionResponses };
};

const normalizeGeminiContentToMessages = (content: GeminiContent): ModelTraceChatMessage[] => {
  const role = content.role === 'model' ? 'assistant' : content.role;
  const { textParts, thinking, toolCalls, functionResponses } = processGeminiContentParts(content.parts);

  // Emit function_response parts as individual tool messages
  const toolMessages = compact(
    functionResponses.map((fr) =>
      prettyPrintChatMessage({
        type: 'message',
        role: 'tool',
        name: fr.name,
        content: JSON.stringify(fr.response),
      }),
    ),
  );

  const messages: ModelTraceChatMessage[] = [...toolMessages];

  // Emit the main message (text and/or tool_calls)
  if (textParts.length > 0 || toolCalls.length > 0) {
    const message = prettyPrintChatMessage({
      type: 'message',
      content: textParts.length > 0 ? textParts : undefined,
      role,
      tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
    });
    if (message) {
      if (thinking) {
        messages.push({ ...message, reasoning: thinking });
      } else {
        messages.push(message);
      }
    }
  } else if (thinking) {
    // Only thinking, no text or tool_calls
    const message = prettyPrintChatMessage({
      type: 'message',
      content: undefined,
      role,
    });
    if (message) {
      messages.push({ ...message, reasoning: thinking });
    }
  }

  return messages;
};

export const normalizeGeminiChatInput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (!isObject(obj)) {
    return null;
  }

  if ('contents' in obj) {
    if (isString(obj.contents)) {
      const message = prettyPrintChatMessage({ type: 'message', content: obj.contents, role: 'user' });
      return message && [message];
    }

    if (isArray(obj.contents) && obj.contents.every(isGeminiContent)) {
      const messages = obj.contents.flatMap(normalizeGeminiContentToMessages);
      return messages.length > 0 ? messages : null;
    }

    // Handle flat list of parts and strings (e.g., [Part.from_bytes(...), "Caption this"])
    if (isArray(obj.contents) && obj.contents.some((item: any) => isGeminiContentPart(item) || isString(item))) {
      const parts: GeminiContentPart[] = [];
      for (const item of obj.contents) {
        if (isString(item)) {
          parts.push({ text: item } as GeminiTextPart);
        } else if (isGeminiContentPart(item)) {
          parts.push(item);
        }
      }
      if (parts.length > 0) {
        const messages = normalizeGeminiContentToMessages({ role: 'user', parts });
        return messages.length > 0 ? messages : null;
      }
    }
  }

  return null;
};

export const normalizeGeminiChatOutput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (!isObject(obj)) {
    return null;
  }

  if ('candidates' in obj && isArray(obj.candidates) && obj.candidates.every(isGeminiCandidate)) {
    const messages = obj.candidates.flatMap((item) => normalizeGeminiContentToMessages(item.content));
    return messages.length > 0 ? messages : null;
  }

  // ADK output format: { content: GeminiContent } without candidates wrapper
  if ('content' in obj && isGeminiContent(obj.content)) {
    const messages = normalizeGeminiContentToMessages(obj.content);
    return messages.length > 0 ? messages : null;
  }

  return null;
};
