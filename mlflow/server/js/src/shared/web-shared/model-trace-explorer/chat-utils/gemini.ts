import { compact, has, isArray, isObject, isString } from 'lodash';

import type { ModelTraceChatMessage, ModelTraceToolCall } from '../ModelTrace.types';
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

type GeminiContentPart = GeminiTextPart | GeminiFunctionCallPart | GeminiFunctionResponsePart;

const isGeminiTextPart = (obj: unknown): obj is GeminiTextPart => {
  return isObject(obj) && 'text' in obj && isString(obj.text);
};

const isGeminiFunctionCallPart = (obj: unknown): obj is GeminiFunctionCallPart => {
  return isObject(obj) && 'function_call' in obj && isObject((obj as any).function_call);
};

const isGeminiFunctionResponsePart = (obj: unknown): obj is GeminiFunctionResponsePart => {
  return isObject(obj) && 'function_response' in obj && isObject((obj as any).function_response);
};

const isGeminiContentPart = (obj: unknown): obj is GeminiContentPart => {
  return isGeminiTextPart(obj) || isGeminiFunctionCallPart(obj) || isGeminiFunctionResponsePart(obj);
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

const isGeminiCandidate = (obj: unknown): obj is GeminiCandidate => {
  return isObject(obj) && 'content' in obj && isGeminiContent(obj.content);
};

const processGeminiContentParts = (
  parts: GeminiContentPart[],
): {
  textParts: { type: 'text'; text: string }[];
  thinking: string | null;
  toolCalls: ModelTraceToolCall[];
  functionResponses: { name: string; response: Record<string, unknown> }[];
} => {
  const textParts: { type: 'text'; text: string }[] = [];
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
    }
  }

  const thinking = thinkingParts.length > 0 ? thinkingParts.join('\n\n') : null;
  return { textParts, thinking, toolCalls, functionResponses };
};

const normalizeGeminiContentToMessages = (content: GeminiContent): ModelTraceChatMessage[] => {
  const role = content.role === 'model' ? 'assistant' : content.role;
  const { textParts, thinking, toolCalls, functionResponses } = processGeminiContentParts(content.parts);

  const messages: ModelTraceChatMessage[] = [];

  // Emit function_response parts as individual tool messages
  for (const fr of functionResponses) {
    const toolMsg = prettyPrintChatMessage({
      type: 'message',
      role: 'tool',
      name: fr.name,
      content: JSON.stringify(fr.response),
    });
    if (toolMsg) {
      messages.push(toolMsg);
    }
  }

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
