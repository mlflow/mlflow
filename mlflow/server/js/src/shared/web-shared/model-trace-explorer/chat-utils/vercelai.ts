import { compact, has, isArray, isObject, isString } from 'lodash';

import type { ModelTraceChatMessage, ModelTraceContentParts, ModelTraceToolCall } from '../ModelTrace.types';
import { prettyPrintChatMessage } from '../ModelTraceExplorer.utils';

type VercelAITextContent = {
  type: 'text';
  text: string;
};

type VercelAIImageContent = {
  type: 'image';
  image: string | URL;
};

type VercelAIToolCall = {
  type: 'tool-call';
  toolCallId: string;
  toolName: string;
  input: string;
  providerOptions: Record<string, any>;
};

type VercelAIToolCallResult = {
  type: 'tool-result';
  toolCallId: string;
  toolName: string;
  output: any;
};

type VercelAIContentPart = VercelAITextContent | VercelAIImageContent | VercelAIToolCall | VercelAIToolCallResult;

export type VercelAIMessage = {
  role: 'user' | 'assistant' | 'system' | 'tool' | 'function';
  content: string | VercelAIContentPart[];
  name?: string;
};

const isVercelAIContentPart = (obj: unknown): obj is VercelAIContentPart => {
  if (!isObject(obj) || !has(obj, 'type')) {
    return false;
  }

  const typedObj = obj as any;

  if (typedObj.type === 'text' && has(obj, 'text') && isString(typedObj.text)) {
    return true;
  }

  if (typedObj.type === 'image' && has(obj, 'image')) {
    return isString(typedObj.image);
  }

  if (isVercelAIToolCall(obj)) return true;

  if (typedObj.type === 'tool-result' && has(obj, 'output')) {
    return true;
  }

  return false;
};

const isVercelAIToolCall = (obj: unknown): obj is VercelAIToolCall => {
  if (!isObject(obj)) {
    return false;
  }

  return has(obj, 'toolCallId') && has(obj, 'toolName') && has(obj, 'input');
};

const isVercelAIMessage = (obj: unknown): obj is VercelAIMessage => {
  if (!isObject(obj)) {
    return false;
  }

  const typedObj = obj as any;

  const hasRole =
    has(obj, 'role') &&
    isString(typedObj.role) &&
    ['user', 'assistant', 'system', 'tool', 'function'].includes(typedObj.role);

  if (!hasRole) {
    return false;
  }

  const hasContent =
    has(obj, 'content') &&
    (isString(typedObj.content) || (isArray(typedObj.content) && typedObj.content.every(isVercelAIContentPart)));

  return hasContent;
};

const normalizeVercelAIContentPart = (item: VercelAIContentPart): ModelTraceContentParts => {
  switch (item.type) {
    case 'text': {
      return { type: 'text', text: item.text };
    }
    case 'image': {
      return { type: 'image_url', image_url: { url: item.image as string } };
    }
    case 'tool-call': {
      return { type: 'text', text: '' };
    }
    case 'tool-result': {
      return { type: 'text', text: JSON.stringify(item.output) };
    }
  }
};

const extractToolCalls = (content: VercelAIContentPart[]): ModelTraceToolCall[] => {
  return content.filter((item) => item.type === 'tool-call').map(processVercelAIToolCall);
};

const processVercelAIMessage = (message: VercelAIMessage): ModelTraceChatMessage | null => {
  if (typeof message.content === 'string') {
    return prettyPrintChatMessage({
      type: 'message',
      content: message.content,
      role: message.role,
      ...(message.name && { name: message.name }),
    });
  } else {
    // Convert content parts array to ModelTraceContentParts
    const contentParts: ModelTraceContentParts[] = message.content.map(normalizeVercelAIContentPart);
    const toolCalls: ModelTraceToolCall[] = extractToolCalls(message.content);
    const toolCallId = message.content.find((item) => item.type === 'tool-result')?.toolCallId;

    return prettyPrintChatMessage({
      content: contentParts,
      role: message.role,
      ...(message.name && { name: message.name }),
      ...(toolCallId && { tool_call_id: toolCallId }),
      ...(toolCalls && toolCalls.length > 0 && { tool_calls: toolCalls }),
    });
  }
};

const processVercelAIToolCall = (toolCall: VercelAIToolCall): ModelTraceToolCall => {
  return {
    id: toolCall.toolCallId,
    function: {
      name: toolCall.toolName,
      arguments: JSON.stringify(toolCall.input),
    },
  };
};
/**
 * Normalize Vercel AI chat input format (generateText.doGenerate)
 */
export const normalizeVercelAIChatInput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (!isObject(obj)) {
    return null;
  }

  const typedObj = obj as any;

  if (has(obj, 'messages') && isArray(typedObj.messages) && typedObj.messages.every(isVercelAIMessage)) {
    return compact(typedObj.messages.map(processVercelAIMessage));
  }

  return null;
};

/**
 * Normalize Vercel AI chat output format
 **/
export const normalizeVercelAIChatOutput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (!isObject(obj)) {
    return null;
  }

  const typedObj = obj as any;

  if (has(obj, 'text') && isString(typedObj.text)) {
    return compact([
      prettyPrintChatMessage({
        type: 'message',
        content: typedObj.text,
        role: 'assistant',
      }),
    ]);
  }

  if (has(obj, 'toolCalls') && isArray(typedObj.toolCalls) && typedObj.toolCalls.every(isVercelAIToolCall)) {
    return compact([
      prettyPrintChatMessage({
        type: 'message',
        content: '',
        role: 'assistant',
        tool_calls: compact(typedObj.toolCalls.map(processVercelAIToolCall)),
      }),
    ]);
  }

  return null;
};
