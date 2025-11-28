import { has, isArray, isNil, isObject, isString } from 'lodash';

import type { ModelTraceChatMessage, ModelTraceToolCall } from '../ModelTrace.types';
import { prettyPrintChatMessage } from '../ModelTraceExplorer.utils';

// Request part types
type SystemPromptPart = {
  part_kind: 'system-prompt';
  content: string;
};

type UserPromptPart = {
  part_kind: 'user-prompt';
  content: string;
};

type ToolReturnPart = {
  part_kind: 'tool-return';
  content: any;
  tool_call_id: string;
};

type BuiltinToolReturnPart = {
  part_kind: 'builtin-tool-return';
  content: any;
  tool_call_id: string;
};

type ModelRequestPart = SystemPromptPart | UserPromptPart | ToolReturnPart | BuiltinToolReturnPart;

// Response part types
type TextPart = {
  part_kind: 'text';
  content: string;
};

type ThinkingPart = {
  part_kind: 'thinking';
  content: string;
};

type ToolCallPart = {
  part_kind: 'tool-call';
  tool_name: string;
  args: string | { [key: string]: any } | null;
  tool_call_id: string;
};

type BuiltinToolCallPart = {
  part_kind: 'builtin-tool-call';
  tool_name: string;
  args: string | { [key: string]: any } | null;
  tool_call_id: string;
};

type FilePart = {
  part_kind: 'file';
};

type ModelResponsePart = TextPart | ThinkingPart | ToolCallPart | BuiltinToolCallPart | FilePart;

// Message types
export type PydanticAIModelRequest = {
  kind: 'request';
  parts: ModelRequestPart[];
};

export type PydanticAIModelResponse = {
  kind: 'response';
  parts: ModelResponsePart[];
};

export type PydanticAIMessage = PydanticAIModelRequest | PydanticAIModelResponse;

// Type guards
const isPydanticAIModelRequest = (obj: any): obj is PydanticAIModelRequest => {
  return isObject(obj) && has(obj, 'kind') && obj.kind === 'request' && has(obj, 'parts') && isArray(obj.parts);
};

const isPydanticAIModelResponse = (obj: any): obj is PydanticAIModelResponse => {
  return isObject(obj) && has(obj, 'kind') && obj.kind === 'response' && has(obj, 'parts') && isArray(obj.parts);
};

// Normalization helpers
const normalizeToolCallArgs = (args: string | { [key: string]: any } | null): string => {
  if (isNil(args)) {
    return '{}';
  }
  if (isString(args)) {
    return args;
  }
  return JSON.stringify(args, null, 2);
};

const normalizePydanticAIToolCall = (part: ToolCallPart | BuiltinToolCallPart): ModelTraceToolCall => {
  return {
    id: part.tool_call_id,
    function: {
      name: part.tool_name,
      arguments: normalizeToolCallArgs(part.args),
    },
  };
};

const normalizeModelRequest = (request: PydanticAIModelRequest): ModelTraceChatMessage[] => {
  const messages: ModelTraceChatMessage[] = [];

  for (const part of request.parts) {
    switch (part.part_kind) {
      case 'system-prompt': {
        const message = prettyPrintChatMessage({
          role: 'system',
          content: part.content,
        });
        if (message) messages.push(message);
        break;
      }
      case 'user-prompt': {
        const message = prettyPrintChatMessage({
          role: 'user',
          content: part.content,
        });
        if (message) messages.push(message);
        break;
      }
      case 'tool-return':
      case 'builtin-tool-return': {
        const contentStr = isString(part.content) ? part.content : JSON.stringify(part.content, null, 2);
        const message = prettyPrintChatMessage({
          role: 'tool',
          content: contentStr,
          tool_call_id: part.tool_call_id,
        });
        if (message) messages.push(message);
        break;
      }
    }
  }

  return messages;
};

const normalizeModelResponse = (response: PydanticAIModelResponse): ModelTraceChatMessage[] => {
  const messages: ModelTraceChatMessage[] = [];
  const textParts: string[] = [];
  const toolCalls: ModelTraceToolCall[] = [];

  for (const part of response.parts) {
    switch (part.part_kind) {
      case 'text': {
        textParts.push(part.content);
        break;
      }
      case 'thinking': {
        textParts.push(`[Thinking] ${part.content}`);
        break;
      }
      case 'tool-call':
      case 'builtin-tool-call': {
        toolCalls.push(normalizePydanticAIToolCall(part));
        break;
      }
      case 'file': {
        textParts.push('[file]');
        break;
      }
    }
  }

  const content = textParts.length > 0 ? textParts.join('\n\n') : undefined;
  const message = prettyPrintChatMessage({
    role: 'assistant',
    content,
    ...(toolCalls.length > 0 && { tool_calls: toolCalls }),
  });

  if (message) {
    messages.push(message);
  }

  return messages;
};

export const normalizePydanticAIChatInput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (!obj) {
    return null;
  }

  if (isObject(obj) && has(obj, 'message_history')) {
    const messageHistory = (obj as any).message_history;
    if (isArray(messageHistory) && messageHistory.length > 0) {
      const messages: ModelTraceChatMessage[] = [];

      for (const item of messageHistory) {
        if (isPydanticAIModelRequest(item)) {
          messages.push(...normalizeModelRequest(item));
        } else if (isPydanticAIModelResponse(item)) {
          messages.push(...normalizeModelResponse(item));
        }
      }

      return messages.length > 0 ? messages : null;
    }
  }

  if (isArray(obj) && obj.length > 0) {
    const messages: ModelTraceChatMessage[] = [];

    for (const item of obj) {
      if (isPydanticAIModelRequest(item)) {
        messages.push(...normalizeModelRequest(item));
      } else if (isPydanticAIModelResponse(item)) {
        messages.push(...normalizeModelResponse(item));
      }
    }

    return messages.length > 0 ? messages : null;
  }

  if (isPydanticAIModelRequest(obj)) {
    return normalizeModelRequest(obj);
  }

  return null;
};

export const normalizePydanticAIChatOutput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (!obj) {
    return null;
  }

  if (isObject(obj) && has(obj, '_new_messages_serialized')) {
    const newMessages = (obj as any)._new_messages_serialized;
    if (isArray(newMessages) && newMessages.length > 0) {
      const messages: ModelTraceChatMessage[] = [];

      for (const item of newMessages) {
        if (isPydanticAIModelRequest(item)) {
          messages.push(...normalizeModelRequest(item));
        } else if (isPydanticAIModelResponse(item)) {
          messages.push(...normalizeModelResponse(item));
        }
      }

      return messages.length > 0 ? messages : null;
    }
  }

  if (isArray(obj) && obj.length > 0) {
    const messages: ModelTraceChatMessage[] = [];

    for (const item of obj) {
      if (isPydanticAIModelRequest(item)) {
        messages.push(...normalizeModelRequest(item));
      } else if (isPydanticAIModelResponse(item)) {
        messages.push(...normalizeModelResponse(item));
      }
    }

    return messages.length > 0 ? messages : null;
  }

  if (isPydanticAIModelResponse(obj)) {
    return normalizeModelResponse(obj);
  }

  return null;
};
