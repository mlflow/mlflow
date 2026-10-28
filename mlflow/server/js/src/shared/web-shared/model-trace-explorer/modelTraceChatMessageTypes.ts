import { has, isArray, isNil, isString } from 'lodash';

import type {
  ModelTraceChatMessage,
  ModelTraceChatResponse,
  ModelTraceToolCall,
  RawModelTraceChatMessage,
} from './ModelTrace.types';

export const isModelTraceToolCall = (obj: any): obj is ModelTraceToolCall => {
  return obj && isString(obj.id) && isString(obj.function?.arguments) && isString(obj.function?.name);
};

const isContentPart = (part: any) => {
  switch (part.type) {
    case 'text':
    case 'input_text':
    case 'output_text':
      return isString(part.text);
    case 'image_url':
      const { image_url } = part;
      if (isNil(image_url)) {
        return false;
      }
      return isString(image_url.url) && (isNil(image_url.detail) || ['auto', 'low', 'high'].includes(image_url.detail));
    case 'input_audio':
      const { input_audio } = part;
      if (isNil(input_audio)) {
        return false;
      }
      return isString(input_audio.data) && (isNil(input_audio.format) || ['wav', 'mp3'].includes(input_audio.format));
    case 'image':
      // Anthropic format: {"type": "image", "source": {"type": "base64", "data": "..."}}
      return !isNil(part.source) && isString((part as any).source.data);
    default:
      return false;
  }
};

const isContentType = (content: any) => {
  if (isNil(content) || isString(content)) {
    return true;
  }

  if (isArray(content)) {
    return content.every((part) => isContentPart(part));
  }

  return false;
};

export const isModelTraceChatMessage = (message: any): message is ModelTraceChatMessage => {
  if (!isRawModelTraceChatMessage(message)) {
    return false;
  }

  return isNil(message.content) || isString(message.content);
};

export const isRawModelTraceChatMessage = (message: any): message is RawModelTraceChatMessage => {
  if (!message) {
    return false;
  }

  if (message.tool_calls) {
    if (!Array.isArray(message.tool_calls)) {
      return false;
    }

    if (!message.tool_calls.every(isModelTraceToolCall)) {
      return false;
    }
  }

  if (message.parts && isNil(message.content)) {
    // OpenTelemetry GenAI semantic conventions are parsed separately.
    return false;
  }

  if (message.type === 'reasoning') {
    return true;
  }

  if (!isContentType(message.content)) {
    return false;
  }

  return (
    message.role === 'user' || message.role === 'assistant' || message.role === 'system' || message.role === 'tool'
  );
};

export const isModelTraceChoices = (obj: any): obj is ModelTraceChatResponse['choices'] => {
  return (
    Array.isArray(obj) &&
    obj.length > 0 &&
    obj.every((choice: any) => has(choice, 'message') && isRawModelTraceChatMessage(choice.message))
  );
};

export const isModelTraceChatResponse = (obj: any): obj is ModelTraceChatResponse => {
  return obj && isModelTraceChoices(obj.choices);
};
