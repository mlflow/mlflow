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
  const isObject = typeof obj === 'object' && obj !== null;
  if (!isObject) {
    return false;
  }

  if ('type' in obj) {
    if (obj.type === 'text' && 'text' in obj && typeof obj.text === 'string') {
      return true;
    }

    if (
      obj.type === 'image' &&
      'source' in obj &&
      typeof obj.source === 'object' &&
      obj.source !== null &&
      'type' in obj.source
    ) {
      if (
        obj.source.type === 'base64' &&
        'media_type' in obj.source &&
        typeof obj.source.media_type === 'string' &&
        ['image/jpeg', 'image/png', 'image/gif', 'image/webp'].includes(obj.source.media_type) &&
        'data' in obj.source &&
        typeof obj.source.data === 'string'
      ) {
        return true;
      }

      if (obj.source.type === 'url' && 'url' in obj.source && typeof obj.source.url === 'string') {
        return true;
      }
    }
  }
  return false;
};

const isAnthropicMessageParam = (obj: unknown): obj is AnthropicMessageParam => {
  const isObject = typeof obj === 'object' && obj !== null;
  if (!isObject) {
    return false;
  }

  const hasRole = 'role' in obj && typeof obj.role === 'string' && ['user', 'assistant'].includes(obj.role);
  const hasContent =
    'content' in obj &&
    (typeof obj.content === 'string' ||
      (Array.isArray(obj.content) && obj.content.every(isAnthropicContentBlockParam)));

  return hasRole && hasContent;
};

const normalizeAnthropicContentBlockParam = (item: AnthropicContentBlockParam): ModelTraceContentParts => {
  return item.type === 'text'
    ? { type: 'text', text: item.text }
    : item.source.type === 'url'
    ? { type: 'image_url', image_url: { url: item.source.url } }
    : {
        type: 'image_url',
        image_url: { url: `data:${item.source.media_type};base64,${item.source.data}` },
      };
};

export const normalizeAnthropicChatInput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (!obj) {
    return null;
  }

  if (
    typeof obj === 'object' &&
    'messages' in obj &&
    Array.isArray(obj.messages) &&
    obj.messages.every(isAnthropicMessageParam)
  ) {
    return obj.messages
      .map((message) =>
        prettyPrintChatMessage({
          type: 'message',
          content:
            typeof message.content === 'string'
              ? message.content
              : message.content.map(normalizeAnthropicContentBlockParam),
          role: message.role,
        }),
      )
      .filter((item) => item !== null);
  }

  return null;
};

export const normalizeAnthropicChatOutput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (!obj) {
    return null;
  }

  if (typeof obj === 'object' && 'type' in obj && obj.type === 'message' && isAnthropicMessageParam(obj)) {
    const message = prettyPrintChatMessage({
      type: 'message',
      content: typeof obj.content === 'string' ? obj.content : obj.content.map(normalizeAnthropicContentBlockParam),
      role: obj.role,
    });
    return message && [message];
  }

  return null;
};
