import { has } from 'lodash';
import { ModelTraceChatMessage } from '../ModelTrace.types';
import { isModelTraceChatMessage } from '../ModelTraceExplorer.utils';

export type LlamaIndexChatResponse = {
  message: ModelTraceChatMessage;
};

export const isLlamaIndexChatResponse = (obj: unknown): obj is LlamaIndexChatResponse => {
  return has(obj, 'message') && isModelTraceChatMessage(obj.message);
};

export const normalizeLlamaIndexChatResponse = (obj: any): ModelTraceChatMessage[] | null => {
  if (!isLlamaIndexChatResponse(obj)) {
    return null;
  }

  return [obj.message];
};
