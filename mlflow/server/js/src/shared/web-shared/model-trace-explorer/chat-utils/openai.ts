import { compact, get, has, isArray, isNil, isObject, isString } from 'lodash';
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

const isOpenAIResponsesInputMessage = (obj: unknown): obj is OpenAIResponsesInputMessage => {
  if (has(obj, 'role') && has(obj, 'content') && ['user', 'assistant', 'system', 'developer'].includes(obj.role)) {
    return (
      isString(obj.content) ||
      (isArray(obj.content) &&
        obj.content.every(
          (item: unknown) => has(item, 'type') && ['input_text', 'input_image', 'input_file'].includes(item.type),
        ))
    );
  }

  return false;
};

export const isOpenAIResponsesInput = (obj: unknown): obj is OpenAIResponsesInput => {
  return isString(obj) || isOpenAIResponsesInputMessage(obj);
};

export const isOpenAIResponsesOutputItem = (obj: unknown): obj is OpenAIResponsesOutputItem => {
  if (!isObject(obj)) {
    return false;
  }

  if (get(obj, 'type') === 'message') {
    return isRawModelTraceChatMessage(obj);
  }

  if (get(obj, 'type') === 'function_call') {
    return isString(get(obj, 'call_id')) && isString(get(obj, 'name')) && isString(get(obj, 'arguments'));
  }

  if (get(obj, 'type') === 'function_call_output') {
    return isString(get(obj, 'call_id')) && isString(get(obj, 'output'));
  }

  if (get(obj, 'type') === 'image_generation_call') {
    const outputFormat = get(obj, 'output_format');
    return isString(get(obj, 'result')) && isString(outputFormat) && ['png', 'jpeg', 'webp'].includes(outputFormat);
  }

  return false;
};

const normalizeOpenAIResponsesInputItem = (
  obj: OpenAIResponsesInputText | OpenAIResponsesInputFile | OpenAIResponsesInputImage,
  role: OpenAIResponsesInputMessageRole,
): ModelTraceChatMessage | null => {
  const text = get(obj, 'text');
  if (get(obj, 'type') === 'input_text' && isString(text)) {
    return prettyPrintChatMessage({
      type: 'message',
      content: [{ type: 'text', text }],
      role: role,
    });
  }

  const imageUrl = get(obj, 'image_url');
  if (get(obj, 'type') === 'input_image' && isString(imageUrl)) {
    return prettyPrintChatMessage({
      type: 'message',
      content: [{ type: 'image_url', image_url: { url: imageUrl } }],
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
  if (isString(obj.content)) {
    const message = prettyPrintChatMessage({ type: 'message', content: obj.content, role: obj.role });
    return message && [message];
  } else {
    return obj.content.map((item) => normalizeOpenAIResponsesInputItem(item, obj.role)).filter((item) => item !== null);
  }
};

export const normalizeOpenAIResponsesInput = (obj: unknown): ModelTraceChatMessage[] | null => {
  const input: unknown = get(obj, 'input');

  if (isString(input)) {
    const message = prettyPrintChatMessage({ type: 'message', content: input, role: 'user' });
    return message && [message];
  }

  if (isArray(input) && input.every(isOpenAIResponsesInputMessage)) {
    return compact(input.flatMap(normalizeOpenAIResponsesInputMessage));
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

  if (obj.type === 'image_generation_call') {
    return prettyPrintChatMessage({
      type: 'message',
      content: [{ type: 'image_url', image_url: { url: `data:image/${obj.output_format};base64,${obj.result}` } }],
      role: 'tool',
    });
  }

  return null;
};

export const normalizeOpenAIResponsesOutput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (isNil(obj)) {
    return null;
  }

  const output: unknown = get(obj, 'output');

  // list of output items
  if (isArray(output) && output.length > 0 && output.every(isOpenAIResponsesOutputItem)) {
    return compact(output.map(normalizeOpenAIResponsesOutputItem));
  }

  // list of output chunks
  if (
    isArray(output) &&
    output.length > 0 &&
    output.every((chunk) => chunk.type === 'response.output_item.done' && isOpenAIResponsesOutputItem(chunk.item))
  ) {
    return compact(output.map((chunk) => normalizeOpenAIResponsesOutputItem(chunk.item)));
  }

  return null;
};
