import { compact, get, has, isArray, isNil, isObject, isString } from 'lodash';

import type { ModelTraceChatMessage } from '../ModelTrace.types';
import { prettyPrintChatMessage, prettyPrintToolCall } from '../ModelTraceExplorer.utils';

interface AutogenFunctionCall {
  id: string;
  name: string;
  arguments: string;
}

interface AutogenSystemMessage {
  source: 'system';
  content: string;
}

interface AutogenUserMessage {
  source: 'user';
  content: string | any[];
}

interface AutogenAssistantMessage {
  source: 'assistant';
  content: string | AutogenFunctionCall[];
}

interface AutogenFunctionExecutionResultMessage {
  source: 'function';
  content: any;
}

type AutogenMessage =
  | AutogenSystemMessage
  | AutogenUserMessage
  | AutogenAssistantMessage
  | AutogenFunctionExecutionResultMessage;

const isAutogenFunctionCall = (obj: unknown): obj is AutogenFunctionCall => {
  return isObject(obj) && isString(get(obj, 'id')) && isString(get(obj, 'name')) && isString(get(obj, 'arguments'));
};

const isAutogenMessage = (obj: unknown): obj is AutogenMessage => {
  if (!isObject(obj)) {
    return false;
  }

  // Handle messages with 'type' field (new format)
  const messageType = get(obj, 'type');
  if (messageType) {
    if (messageType === 'SystemMessage') {
      return isString(get(obj, 'content'));
    }
    if (messageType === 'UserMessage') {
      return has(obj, 'content') && has(obj, 'source');
    }
    if (messageType === 'AssistantMessage') {
      return has(obj, 'content') && has(obj, 'source');
    }
    if (messageType === 'FunctionMessage') {
      return has(obj, 'content') && has(obj, 'source');
    }
  }

  // Handle messages with 'source' field (old format)
  if (has(obj, 'source') && ['system', 'user', 'assistant', 'function'].includes(get(obj, 'source'))) {
    return has(obj, 'content');
  }

  return false;
};

const convertAssistantMessageToChatMessage = (
  content: string | AutogenFunctionCall[],
): ModelTraceChatMessage | null => {
  if (isString(content)) {
    return prettyPrintChatMessage({ type: 'message', content, role: 'assistant' });
  }

  if (isArray(content) && content.every(isAutogenFunctionCall)) {
    return {
      role: 'assistant',
      tool_calls: content.map((f) =>
        prettyPrintToolCall({
          id: f.id,
          function: {
            name: f.name,
            arguments: f.arguments,
          },
        }),
      ),
    };
  }

  return null;
};

const normalizeAutogenMessage = (message: any): ModelTraceChatMessage | null => {
  // Handle messages with 'type' field (new format)
  if (message.type === 'SystemMessage') {
    return prettyPrintChatMessage({ type: 'message', content: message.content, role: 'system' });
  }

  if (message.type === 'UserMessage') {
    if (isString(message.content)) {
      return prettyPrintChatMessage({ type: 'message', content: message.content, role: 'user' });
    }

    if (isArray(message.content)) {
      // Handle content that might be an array of text/image parts
      const textParts = message.content
        .filter((part: any) => isString(part) || (isObject(part) && (part as any).type === 'text'))
        .map((part: any) => (isString(part) ? { type: 'text' as const, text: part } : part));

      if (textParts.length > 0) {
        return prettyPrintChatMessage({ type: 'message', content: textParts, role: 'user' });
      }
    }
  }

  if (message.type === 'AssistantMessage') {
    return convertAssistantMessageToChatMessage(message.content);
  }

  if (message.type === 'FunctionMessage') {
    // Function execution result messages are typically logged as user messages
    return prettyPrintChatMessage({ type: 'message', content: JSON.stringify(message.content), role: 'user' });
  }

  // Handle messages with 'source' field (old format)
  if (message.source === 'system') {
    return prettyPrintChatMessage({ type: 'message', content: message.content, role: 'system' });
  }

  if (message.source === 'user') {
    if (isString(message.content)) {
      return prettyPrintChatMessage({ type: 'message', content: message.content, role: 'user' });
    }

    if (isArray(message.content)) {
      // Handle content that might be an array of text/image parts
      const textParts = message.content
        .filter((part: any) => isString(part))
        .map((part: any) => ({ type: 'text' as const, text: part }));

      if (textParts.length > 0) {
        return prettyPrintChatMessage({ type: 'message', content: textParts, role: 'user' });
      }
    }
  }

  if (message.source === 'assistant') {
    return convertAssistantMessageToChatMessage(message.content);
  }

  if (message.source === 'function') {
    // Function execution result messages are typically logged as user messages
    return prettyPrintChatMessage({ type: 'message', content: JSON.stringify(message.content), role: 'user' });
  }

  return null;
};

export const normalizeAutogenChatInput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (isNil(obj)) {
    return null;
  }

  // Handle case where input is directly an array of messages
  if (isArray(obj) && obj.length > 0 && obj.every(isAutogenMessage)) {
    return compact(obj.map(normalizeAutogenMessage));
  }

  // Handle case where input is wrapped in an object with 'messages' key
  if (isObject(obj) && 'messages' in obj) {
    const messages = (obj as any).messages;
    if (isArray(messages) && messages.length > 0 && messages.every(isAutogenMessage)) {
      return compact(messages.map(normalizeAutogenMessage));
    }
  }

  return null;
};

export const normalizeAutogenChatOutput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (isNil(obj)) {
    return null;
  }

  // Handle case where output is directly an array of messages
  if (isArray(obj) && obj.length > 0 && obj.every(isAutogenMessage)) {
    return compact(obj.map(normalizeAutogenMessage));
  }

  // Handle case where output is wrapped in an object with 'messages' key
  if (isObject(obj) && 'content' in obj) {
    const message = prettyPrintChatMessage({ role: 'assistant', content: obj.content as string, type: 'message' });
    return message ? [message] : null;
  }

  return null;
};
