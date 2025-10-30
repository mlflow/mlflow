import { compact, has, isArray, isNil, isObject, isString } from 'lodash';

import type { ModelTraceChatMessage, ModelTraceContentParts } from '../ModelTrace.types';
import { prettyPrintChatMessage } from '../ModelTraceExplorer.utils';

/**
 * Vercel AI SDK Input format:
 * 
 * Format 1 (with messages):
 * {
 *   messages: [
 *     {
 *       role: "user" | "assistant" | "system" | "tool",
 *       content: [
 *         { type: "text", text: "..." },
 *         { type: "image", image: "..." },
 *         ...
 *       ]
 *     }
 *   ]
 * }
 * 
 * Format 2 (with prompt):
 * {
 *   prompt: "..."
 * }
 */

type VercelAITextContent = {
  type: 'text';
  text: string;
};

type VercelAIImageContent = {
  type: 'image';
  image: string | URL;
};

type VercelAIContentPart = VercelAITextContent | VercelAIImageContent;

type VercelAIMessage = {
  role: 'user' | 'assistant' | 'system' | 'tool' | 'function';
  content: string | VercelAIContentPart[];
  name?: string;
  tool_call_id?: string;
};

type VercelAIMessagesInput = {
  messages: VercelAIMessage[];
};

type VercelAIPromptInput = {
  prompt: string;
};

/**
 * Vercel AI SDK Output format:
 * 
 * {
 *   text?: string,
 *   response?: {
 *     messages?: [...],
 *     text?: string
 *   },
 *   ...
 * }
 */
type VercelAIOutput = {
  text?: string;
  response?: {
    messages?: VercelAIMessage[];
    text?: string;
  };
  messages?: VercelAIMessage[];
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
    return isString(typedObj.image) || typedObj.image instanceof URL;
  }

  return false;
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
      const imageUrl = item.image instanceof URL ? item.image.toString() : item.image;
      return { type: 'image_url', image_url: { url: imageUrl } };
    }
  }
};

const processVercelAIMessage = (message: VercelAIMessage): ModelTraceChatMessage | null => {
  if (typeof message.content === 'string') {
    return prettyPrintChatMessage({
      type: 'message',
      content: message.content,
      role: message.role,
      ...(message.name && { name: message.name }),
      ...(message.tool_call_id && { tool_call_id: message.tool_call_id }),
    });
  } else {
    // Convert content parts array to ModelTraceContentParts
    const contentParts: ModelTraceContentParts[] = message.content.map(normalizeVercelAIContentPart);

    return prettyPrintChatMessage({
      type: 'message',
      content: contentParts,
      role: message.role,
      ...(message.name && { name: message.name }),
      ...(message.tool_call_id && { tool_call_id: message.tool_call_id }),
    });
  }
};

/**
 * Normalize Vercel AI chat input format
 * 
 * Handles two formats:
 * 1. { messages: [...] }
 * 2. { prompt: "..." }
 */
export const normalizeVercelAIChatInput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (!isObject(obj)) {
    return null;
  }

  const typedObj = obj as any;

  // Format 1: messages array
  if (has(obj, 'messages') && isArray(typedObj.messages) && typedObj.messages.every(isVercelAIMessage)) {
    return compact(typedObj.messages.map(processVercelAIMessage));
  }

  // Format 2: prompt string
  if (has(obj, 'prompt') && isString(typedObj.prompt)) {
    const message = prettyPrintChatMessage({
      type: 'message',
      role: 'user',
      content: typedObj.prompt,
    });
    return message ? [message] : null;
  }

  return null;
};

/**
 * Normalize Vercel AI chat output format
 * 
 * Handles various output formats:
 * 1. { text: "..." }
 * 2. { response: { text: "..." } }
 * 3. { response: { messages: [...] } }
 * 4. { messages: [...] }
 */
export const normalizeVercelAIChatOutput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (!isObject(obj)) {
    return null;
  }

  const typedObj = obj as any;

  // Check for messages in the response
  if (has(obj, 'response.messages') && isArray(typedObj.response.messages) && typedObj.response.messages.every(isVercelAIMessage)) {
    return compact(typedObj.response.messages.map(processVercelAIMessage));
  }

  // Check for messages at the top level
  if (has(obj, 'messages') && isArray(typedObj.messages) && typedObj.messages.every(isVercelAIMessage)) {
    return compact(typedObj.messages.map(processVercelAIMessage));
  }

  // Check for text in response
  if (has(obj, 'response.text') && isString(typedObj.response.text)) {
    const message = prettyPrintChatMessage({
      type: 'message',
      role: 'assistant',
      content: typedObj.response.text,
    });
    return message ? [message] : null;
  }

  // Check for text at top level
  if (has(obj, 'text') && isString(typedObj.text)) {
    const message = prettyPrintChatMessage({
      type: 'message',
      role: 'assistant',
      content: typedObj.text,
    });
    return message ? [message] : null;
  }

  return null;
};

