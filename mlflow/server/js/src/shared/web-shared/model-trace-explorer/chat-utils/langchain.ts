import { compact, has, isNil, isString } from 'lodash';

import type { ModelTraceChatMessage, ModelTraceToolCall } from '../ModelTrace.types';
import { isModelTraceToolCall, prettyPrintToolCall } from '../ModelTraceExplorer.utils';

type LangchainContentPart = {
  type: 'text' | 'image_url';
  text?: string;
  image_url?: {
    url: string;
  };
};

// it has other fields, but we only care about these for now
export type LangchainBaseMessage = {
  content?: string | LangchainContentPart[];
  type: 'human' | 'user' | 'assistant' | 'ai' | 'system' | 'tool' | 'function';
  tool_calls?: LangchainToolCallMessage[];
  tool_call_id?: string;
  additional_kwargs?: {
    // some langchain models have tool_calls specified in additional_kwargs in
    // OpenAI format. this appears to be a bug, but we should still try to handle it
    tool_calls?: ModelTraceToolCall[];
  };
};

export type LangchainToolCallMessage = {
  name: string;
  // an object with the arguments to the tool call.
  // should be stringified before display.
  args: any;
  id: string;
};

export type LangchainChatGeneration = {
  message: LangchainBaseMessage;
};

export const langchainMessageToModelTraceMessage = (message: LangchainBaseMessage): ModelTraceChatMessage | null => {
  let role: ModelTraceChatMessage['role'];
  switch (message.type) {
    case 'user':
    case 'human':
      role = 'user';
      break;
    case 'assistant':
    case 'ai':
      role = 'assistant';
      break;
    case 'system':
      role = 'system';
      break;
    case 'tool':
      role = 'tool';
      break;
    case 'function':
      role = 'function';
      break;
    default:
      return null;
  }

  // Handle content that could be a string or an array of content parts
  let content: string | undefined;
  if (isString(message.content)) {
    content = message.content;
  } else if (Array.isArray(message.content)) {
    // Convert array of content parts to string representation
    const contentParts = message.content
      .map((part: any) => {
        if (isString(part)) {
          return part;
        } else if (part.type === 'text' && part.text) {
          return part.text;
        } else if (part.type === 'image_url' && part.image_url?.url) {
          // Convert to markdown image format with spacing
          return `![](${part.image_url.url})`;
        }
        return '';
      })
      .filter(Boolean);

    // Join with double line breaks for better visual separation
    content = contentParts.join('\n\n');
  } else {
    content = undefined;
  }

  const normalizedMessage: ModelTraceChatMessage = {
    content,
    role,
  };

  const toolCalls = message.tool_calls;
  const toolCallsFromKwargs = message.additional_kwargs?.tool_calls;

  // attempt to parse tool calls from the top-level field,
  // otherwise fall back to the additional_kwargs field if it exists
  if (
    !isNil(toolCalls) &&
    Array.isArray(toolCalls) &&
    toolCalls.length > 0 &&
    toolCalls.every(isLangchainToolCallMessage)
  ) {
    // compact for typing. the coercion should not fail since we
    // check that the type is correct in the if condition above
    normalizedMessage.tool_calls = compact(toolCalls.map(normalizeLangchainToolCall));
  } else if (
    !isNil(toolCallsFromKwargs) &&
    Array.isArray(toolCallsFromKwargs) &&
    toolCallsFromKwargs.length > 0 &&
    toolCallsFromKwargs.every(isModelTraceToolCall)
  ) {
    normalizedMessage.tool_calls = toolCallsFromKwargs.map(prettyPrintToolCall);
  }

  if (!isNil(message.tool_call_id)) {
    normalizedMessage.tool_call_id = message.tool_call_id;
  }

  return normalizedMessage;
};

export const normalizeLangchainToolCall = (toolCall: LangchainToolCallMessage): ModelTraceToolCall | null => {
  return {
    id: toolCall.id,
    function: {
      arguments: JSON.stringify(toolCall.args, null, 2),
      name: toolCall.name,
    },
  };
};

export const isLangchainBaseMessage = (obj: any): obj is LangchainBaseMessage => {
  if (!obj) {
    return false;
  }

  // content can be undefined/null, string, or array of content parts
  if (!isNil(obj.content) && !isString(obj.content) && !Array.isArray(obj.content)) {
    return false;
  }

  // tool call validation is handled by the normalization function
  return ['human', 'user', 'assistant', 'ai', 'system', 'tool', 'function'].includes(obj.type);
};

export const isLangchainToolCallMessage = (obj: any): obj is LangchainToolCallMessage => {
  return obj && isString(obj.name) && has(obj, 'args') && isString(obj.id);
};

export const isLangchainChatGeneration = (obj: any): obj is LangchainChatGeneration => {
  return obj && isLangchainBaseMessage(obj.message);
};

// normalize langchain chat input format
export const normalizeLangchainChatInput = (obj: any): ModelTraceChatMessage[] | null => {
  // it could be a list of list of messages
  if (
    Array.isArray(obj) &&
    obj.length === 1 &&
    Array.isArray(obj[0]) &&
    obj[0].length > 0 &&
    obj[0].every(isLangchainBaseMessage)
  ) {
    const messages = obj[0].map(langchainMessageToModelTraceMessage);
    // if we couldn't convert all the messages, then consider the input invalid
    if (messages.some((message) => message === null)) {
      return null;
    }

    return messages as ModelTraceChatMessage[];
  }

  // it could also be an object with the `messages` key
  if (Array.isArray(obj?.messages) && obj.messages.length > 0 && obj.messages.every(isLangchainBaseMessage)) {
    const messages = obj.messages.map(langchainMessageToModelTraceMessage);

    if (messages.some((message: ModelTraceChatMessage[] | null) => message === null)) {
      return null;
    }

    return messages as ModelTraceChatMessage[];
  }

  // it could also just be a plain array that is in the correct format
  if (Array.isArray(obj) && obj.length > 0 && obj.every(isLangchainBaseMessage)) {
    const messages = obj.map(langchainMessageToModelTraceMessage);

    if (messages.some((message) => message === null)) {
      return null;
    }

    return messages as ModelTraceChatMessage[];
  }

  return null;
};

const isLangchainChatGenerations = (obj: any): obj is LangchainChatGeneration[][] => {
  if (!Array.isArray(obj) || obj.length < 1) {
    return false;
  }

  if (!Array.isArray(obj[0]) || obj[0].length < 1) {
    return false;
  }

  // langchain chat generations are a list of lists of messages
  return obj[0].every(isLangchainChatGeneration);
};

const getMessagesFromLangchainChatGenerations = (
  generations: LangchainChatGeneration[],
): ModelTraceChatMessage[] | null => {
  const messages = generations.map((generation: LangchainChatGeneration) =>
    langchainMessageToModelTraceMessage(generation.message),
  );

  if (messages.some((message) => message === null)) {
    return null;
  }

  return messages as ModelTraceChatMessage[];
};

// detect if an object is a langchain ChatResult, and normalize it to a list of messages
export const normalizeLangchainChatResult = (obj: any): ModelTraceChatMessage[] | null => {
  if (isLangchainChatGenerations(obj)) {
    return getMessagesFromLangchainChatGenerations(obj[0]);
  }

  if (
    !Array.isArray(obj?.generations) ||
    !(obj.generations.length > 0) ||
    !obj.generations[0].every(isLangchainChatGeneration)
  ) {
    return null;
  }

  return getMessagesFromLangchainChatGenerations(obj.generations[0]);
};
