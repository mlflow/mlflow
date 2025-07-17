import { compact, isString } from 'lodash';
import { ModelTraceChatMessage } from '../ModelTrace.types';
import {
  isModelTraceChatResponse,
  isModelTraceChoices,
  isRawModelTraceChatMessage,
  prettyPrintChatMessage,
  prettyPrintToolCall,
} from '../ModelTraceExplorer.utils';
import {
  OpenAIResponsesInput,
  OpenAIResponsesInputFile,
  OpenAIResponsesInputImage,
  OpenAIResponsesInputMessage,
  OpenAIResponsesInputMessageRole,
  OpenAIResponsesInputText,
  OpenAIResponsesOutputItem,
} from '../chat-utils/openai.types';

// normalize the OpenAI chat input format (object with 'messages' or 'input' key)
export const normalizeOpenAIChatInput = (obj: any): ModelTraceChatMessage[] | null => {
  if (!obj) {
    return null;
  }

  const messages = obj.messages ?? obj.input;
  if (!Array.isArray(messages) || messages.length === 0 || !messages.every(isRawModelTraceChatMessage)) {
    return null;
  }

  return compact(messages.map(prettyPrintChatMessage));
};

// normalize the OpenAI chat response format (object with 'choices' key)
export const normalizeOpenAIChatResponse = (obj: any): ModelTraceChatMessage[] | null => {
  if (isModelTraceChoices(obj)) {
    return obj.map((choice) => ({
      ...choice.message,
      tool_calls: choice.message.tool_calls?.map(prettyPrintToolCall),
    }));
  }

  if (!isModelTraceChatResponse(obj)) {
    return null;
  }

  return obj.choices.map((choice) => ({
    ...choice.message,
    tool_calls: choice.message.tool_calls?.map(prettyPrintToolCall),
  }));
};

const isOpenAIResponsesInputMessage = (obj: any): obj is OpenAIResponsesInputMessage => {
  if (['user', 'assistant', 'system', 'developer'].includes(obj.role)) {
    return (
      typeof obj.content === 'string' ||
      (Array.isArray(obj.content) &&
        obj.content.every(
          (item: unknown) =>
            item &&
            typeof item === 'object' &&
            'type' in item &&
            typeof item.type === 'string' &&
            ['input_text', 'input_image', 'input_file'].includes(item.type),
        ))
    );
  }

  return false;
};

export const isOpenAIResponsesInput = (obj: any): obj is OpenAIResponsesInput => {
  if (!obj) {
    return false;
  }

  if (typeof obj === 'string') {
    return true;
  }

  if (isOpenAIResponsesInputMessage(obj)) {
    return true;
  }

  return false;
};

export const isOpenAIResponsesOutputItem = (obj: any): obj is OpenAIResponsesOutputItem => {
  if (!obj) {
    return false;
  }

  if (obj.type === 'message') {
    return isRawModelTraceChatMessage(obj);
  }

  if (obj.type === 'function_call') {
    return isString(obj.call_id) && isString(obj.name) && isString(obj.arguments);
  }

  if (obj.type === 'function_call_output') {
    return isString(obj.call_id) && isString(obj.output);
  }

  return false;
};

const normalizeOpenAIResponsesInputItem = (
  obj: OpenAIResponsesInputText | OpenAIResponsesInputFile | OpenAIResponsesInputImage,
  role: OpenAIResponsesInputMessageRole,
): ModelTraceChatMessage | null => {
  if ('type' in obj && obj.type === 'input_text') {
    return prettyPrintChatMessage({
      type: 'message',
      content: [{ type: 'text', text: obj.text }],
      role: role,
    });
  }

  if ('type' in obj && obj.type === 'input_image' && obj.image_url) {
    return prettyPrintChatMessage({
      type: 'message',
      content: [{ type: 'image_url', image_url: { url: obj.image_url } }],
      role: role,
    });
  }

  // TODO: file input not supported yet
  // if ('type' in obj && obj.type === 'input_file') {
  //   return prettyPrintChatMessage({ type: 'message', content: obj.file_url, role: role });
  // }

  return null;
};

const normalizeOpenAIResponsesInputMessage = (obj: OpenAIResponsesInputMessage): ModelTraceChatMessage[] | null => {
  if (typeof obj.content === 'string') {
    const message = prettyPrintChatMessage({ type: 'message', content: obj.content, role: obj.role });
    return message && [message];
  } else {
    return obj.content.map((item) => normalizeOpenAIResponsesInputItem(item, obj.role)).filter((item) => item !== null);
  }
};

export const normalizeOpenAIResponsesInput = (obj: any): ModelTraceChatMessage[] | null => {
  if (typeof obj === 'string') {
    const message = prettyPrintChatMessage({ type: 'message', content: obj, role: 'user' });
    return message && [message];
  }

  const messages = obj.messages ?? obj.input;
  if (Array.isArray(messages) && messages.every(isOpenAIResponsesInputMessage)) {
    return messages.flatMap(normalizeOpenAIResponsesInputMessage).filter((message) => message !== null);
  }

  return null;
};

export const normalizeOpenAIResponsesOutputItem = (obj: OpenAIResponsesOutputItem): ModelTraceChatMessage | null => {
  if (obj.type === 'message') {
    return prettyPrintChatMessage(obj);
  }

  if (obj.type === 'function_call') {
    return {
      role: 'assistant',
      tool_calls: [
        prettyPrintToolCall({
          id: obj.call_id,
          function: {
            arguments: obj.arguments,
            name: obj.name,
          },
        }),
      ],
    };
  }

  if (obj.type === 'function_call_output') {
    return {
      role: 'tool',
      tool_call_id: obj.call_id,
      content: obj.output,
    };
  }

  return null;
};

export const normalizeOpenAIResponsesOutput = (obj: any): ModelTraceChatMessage[] | null => {
  if (!obj) {
    return null;
  }

  const output = obj.output;

  // list of output items
  if (Array.isArray(output) && output.length > 0 && output.every(isOpenAIResponsesOutputItem)) {
    return compact(output.map(normalizeOpenAIResponsesOutputItem));
  }

  // list of output chunks
  if (
    Array.isArray(output) &&
    output.length > 0 &&
    output.every((chunk) => chunk.type === 'response.output_item.done' && isOpenAIResponsesOutputItem(chunk.item))
  ) {
    return compact(output.map((chunk) => normalizeOpenAIResponsesOutputItem(chunk.item)));
  }

  return null;
};
