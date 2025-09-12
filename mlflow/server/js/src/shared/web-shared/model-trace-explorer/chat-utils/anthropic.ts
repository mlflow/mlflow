import { has, isArray, isNil, isObject, isString } from 'lodash';

import type { ModelTraceChatMessage, ModelTraceContentParts } from '../ModelTrace.types';
import { prettyPrintChatMessage } from '../ModelTraceExplorer.utils';

export type AnthropicMessagesInput = {
  messages: AnthropicMessageParam[];
};

export type AnthropicMessagesOutput = {
  id: string;
  content: AnthropicContentBlock[];
  role: 'assistant';
  type: 'message';
  // model: Model;
  // stop_reason: StopReason | null;
  // stop_sequence: string | null;
  // usage: Usage;
};

type AnthropicContentBlock = AnthropicTextBlock | AnthropicToolUseBlock;
// | ThinkingBlock
// | RedactedThinkingBlock
// | ServerToolUseBlock
// | WebSearchToolResultBlock;

type AnthropicMessageParam = {
  content: string | AnthropicContentBlockParam[];
  role: 'user' | 'assistant';
};

type AnthropicContentBlockParam =
  | AnthropicTextBlockParam
  | AnthropicImageBlockParam
  | AnthropicToolUseBlockParam
  | AnthropicToolResultBlockParam;
// | DocumentBlockParam
// | ThinkingBlockParam
// | RedactedThinkingBlockParam
// | ServerToolUseBlockParam
// | WebSearchToolResultBlockParam;

type AnthropicTextBlockParam = {
  text: string;
  type: 'text';
};

type AnthropicTextBlock = {
  text: string;
  type: 'text';
};

type AnthropicImageBlockParam = {
  source: AnthropicBase64ImageSource | AnthropicURLImageSource;
  type: 'image';
};

type AnthropicBase64ImageSource = {
  type: 'base64';
  data: string;
  media_type: 'image/jpeg' | 'image/png' | 'image/gif' | 'image/webp';
};

type AnthropicURLImageSource = {
  type: 'url';
  url: string;
};

type AnthropicToolUseBlockParam = {
  id: string;
  input: Record<string, any>;
  name: string;
  type: 'tool_use';
};

type AnthropicToolUseBlock = {
  id: string;
  input: Record<string, any>;
  name: string;
  type: 'tool_use';
};

type AnthropicToolResultBlockParam = {
  content: string;
  tool_use_id: string;
  type: 'tool_result';
};

const isAnthropicContentBlockParam = (obj: unknown): obj is AnthropicContentBlockParam => {
  if (isNil(obj)) {
    return false;
  }

  if (has(obj, 'type')) {
    if (obj.type === 'text' && has(obj, 'text') && isString(obj.text)) {
      return true;
    }

    if (obj.type === 'image' && has(obj, 'source') && has(obj.source, 'type')) {
      if (
        obj.source.type === 'base64' &&
        has(obj.source, 'media_type') &&
        isString(obj.source.media_type) &&
        ['image/jpeg', 'image/png', 'image/gif', 'image/webp'].includes(obj.source.media_type) &&
        has(obj.source, 'data') &&
        isString(obj.source.data)
      ) {
        return true;
      }

      if (obj.source.type === 'url' && has(obj.source, 'url') && isString(obj.source.url)) {
        return true;
      }
    }

    if (obj.type === 'tool_use' && has(obj, 'id') && has(obj, 'name') && has(obj, 'input')) {
      return isString(obj.id) && isString(obj.name) && isObject(obj.input);
    }

    if (obj.type === 'tool_result' && has(obj, 'tool_use_id') && has(obj, 'content')) {
      return isString(obj.tool_use_id) && isString(obj.content);
    }
  }
  return false;
};

const isAnthropicMessageParam = (obj: unknown): obj is AnthropicMessageParam => {
  if (!isObject(obj)) {
    return false;
  }

  const hasRole = has(obj, 'role') && isString(obj.role) && ['user', 'assistant'].includes(obj.role);
  const hasContent =
    'content' in obj &&
    (isString(obj.content) || (isArray(obj.content) && obj.content.every(isAnthropicContentBlockParam)));

  return hasRole && hasContent;
};

const normalizeAnthropicContentBlockParam = (item: AnthropicContentBlockParam): ModelTraceContentParts => {
  switch (item.type) {
    case 'text': {
      return { type: 'text', text: item.text };
    }
    case 'image': {
      switch (item.source.type) {
        case 'base64': {
          return {
            type: 'image_url',
            image_url: { url: `data:${item.source.media_type};base64,${item.source.data}` },
          };
        }
        case 'url': {
          return { type: 'image_url', image_url: { url: item.source.url } };
        }
      }
    }
  }
  throw new Error(`Unsupported content block type: ${(item as any).type}`);
};

const processAnthropicMessageContent = (
  content: AnthropicContentBlockParam[],
): {
  messages: ModelTraceChatMessage[];
  textParts: ModelTraceContentParts[];
  toolCalls: any[];
} => {
  const messages: ModelTraceChatMessage[] = [];
  const textParts: ModelTraceContentParts[] = [];
  const toolCalls: any[] = [];

  for (const item of content) {
    if (item.type === 'text' || item.type === 'image') {
      textParts.push(normalizeAnthropicContentBlockParam(item));
    } else if (item.type === 'tool_use') {
      toolCalls.push({
        id: item.id,
        function: {
          name: item.name,
          arguments: JSON.stringify(item.input),
        },
      });
    } else if (item.type === 'tool_result') {
      messages.push({
        role: 'tool',
        tool_call_id: item.tool_use_id,
        content: item.content,
      });
    }
  }

  return { messages, textParts, toolCalls };
};

const processAnthropicMessage = (message: AnthropicMessageParam): ModelTraceChatMessage[] => {
  const messages: ModelTraceChatMessage[] = [];

  if (typeof message.content === 'string') {
    const chatMessage = prettyPrintChatMessage({
      type: 'message',
      content: message.content,
      role: message.role,
    });
    if (chatMessage) messages.push(chatMessage);
  } else {
    const { messages: toolMessages, textParts, toolCalls } = processAnthropicMessageContent(message.content);
    messages.push(...toolMessages);

    if (textParts.length > 0 || toolCalls.length > 0) {
      const chatMessage = prettyPrintChatMessage({
        type: 'message',
        content: textParts.length > 0 ? textParts : undefined,
        role: message.role,
        ...(toolCalls.length > 0 && { tool_calls: toolCalls }),
      });
      if (chatMessage) messages.push(chatMessage);
    }
  }

  return messages;
};

export const normalizeAnthropicChatInput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (!isObject(obj)) {
    return null;
  }

  if ('messages' in obj && isArray(obj.messages) && obj.messages.every(isAnthropicMessageParam)) {
    const messages: ModelTraceChatMessage[] = [];

    for (const message of obj.messages) {
      messages.push(...processAnthropicMessage(message));
    }

    return messages;
  }

  return null;
};

export const normalizeAnthropicChatOutput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (!isObject(obj)) {
    return null;
  }

  if (has(obj, 'type') && obj.type === 'message' && isAnthropicMessageParam(obj)) {
    return processAnthropicMessage(obj);
  }

  return null;
};
