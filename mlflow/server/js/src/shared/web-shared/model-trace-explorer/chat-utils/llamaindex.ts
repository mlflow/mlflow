import { compact, get, isArray, isString } from 'lodash';

import type { ModelTraceChatMessage } from '../ModelTrace.types';
import { prettyPrintChatMessage } from '../ModelTraceExplorer.utils';

export type LlamaIndexChatResponse = {
  message: LlamaIndexChatMessage;
};

type LlamaIndexChatMessage = {
  role: 'assistant' | 'system' | 'user';
  blocks: LlamaIndexChatMessageBlock[];
};

type LlamaIndexChatMessageBlock = {
  block_type: 'text';
  text: 'string';
};

export type LlamaIndexChatInput = {
  messages: LlamaIndexChatMessage[];
};

const isLlamaIndexChatMessageBlock = (obj: unknown): obj is LlamaIndexChatMessageBlock => {
  const text = get(obj, 'text');
  return get(obj, 'block_type') === 'text' && isString(text);
};

const isLlamaIndexChatMessage = (message: unknown): message is LlamaIndexChatMessage => {
  const blocks: unknown = get(message, 'blocks');
  const role: unknown = get(message, 'role');
  return (
    isString(role) &&
    ['assistant', 'system', 'user'].includes(role) &&
    isArray(blocks) &&
    blocks.every(isLlamaIndexChatMessageBlock)
  );
};

const isLlamaIndexChatResponse = (obj: unknown): obj is LlamaIndexChatResponse => {
  return isLlamaIndexChatMessage(get(obj, 'message'));
};

const isLlamaIndexChatInput = (obj: unknown): obj is LlamaIndexChatInput => {
  const messages: unknown = get(obj, 'messages');
  return isArray(messages) && messages.every(isLlamaIndexChatMessage);
};

const prettyPrintMessage = (message: LlamaIndexChatMessage) => {
  return prettyPrintChatMessage({
    role: message.role,
    content: message.blocks.map((block) => ({ type: 'text', text: block.text })),
  });
};

export const normalizeLlamaIndexChatResponse = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (!isLlamaIndexChatResponse(obj)) {
    return null;
  }

  const message = prettyPrintMessage(obj.message);
  return message && [message];
};

export const normalizeLlamaIndexChatInput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (!isLlamaIndexChatInput(obj)) {
    return null;
  }

  return compact(obj.messages.map(prettyPrintMessage));
};
