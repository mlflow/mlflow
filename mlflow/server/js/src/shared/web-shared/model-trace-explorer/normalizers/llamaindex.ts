import { ModelTraceChatMessage } from '../ModelTrace.types';
import { isModelTraceChatMessage } from '../ModelTraceExplorer.utils';

export type LlamaIndexChatResponse = {
  message: ModelTraceChatMessage;
};

export const isLlamaIndexChatResponse = (obj: any): obj is LlamaIndexChatResponse => {
  return obj && isModelTraceChatMessage(obj.message);
};

export const normalizeLlamaIndexChatResponse = (obj: any): ModelTraceChatMessage[] | null => {
  if (!isLlamaIndexChatResponse(obj)) {
    return null;
  }

  return [obj.message];
};
