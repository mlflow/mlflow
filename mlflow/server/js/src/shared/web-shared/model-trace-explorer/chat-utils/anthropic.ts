import { compact, has, isArray, isNil, isObject, isString } from 'lodash';
import { ModelTraceChatMessage, ModelTraceContentParts } from '../ModelTrace.types';
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

type AnthropicContentBlock = AnthropicTextBlock;
// | ThinkingBlock
// | RedactedThinkingBlock
// | ToolUseBlock
// | ServerToolUseBlock
// | WebSearchToolResultBlock;

type AnthropicMessageParam = {
  content: string | AnthropicContentBlockParam[];
  role: 'user' | 'assistant';
};

type AnthropicContentBlockParam = AnthropicTextBlockParam | AnthropicImageBlockParam;
// | DocumentBlockParam
// | ThinkingBlockParam
// | RedactedThinkingBlockParam
// | ToolUseBlockParam
// | ToolResultBlockParam
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
};

export const normalizeAnthropicChatInput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (!isObject(obj)) {
    return null;
  }

  if ('messages' in obj && isArray(obj.messages) && obj.messages.every(isAnthropicMessageParam)) {
    return compact(
      obj.messages.map((message) =>
        prettyPrintChatMessage({
          type: 'message',
          content:
            typeof message.content === 'string'
              ? message.content
              : message.content.map(normalizeAnthropicContentBlockParam),
          role: message.role,
        }),
      ),
    );
  }

  return null;
};

export const normalizeAnthropicChatOutput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (!isObject(obj)) {
    return null;
  }

  if (has(obj, 'type') && obj.type === 'message' && isAnthropicMessageParam(obj)) {
    const message = prettyPrintChatMessage({
      type: 'message',
      content: typeof obj.content === 'string' ? obj.content : obj.content.map(normalizeAnthropicContentBlockParam),
      role: obj.role,
    });
    return message && [message];
  }

  return null;
};
