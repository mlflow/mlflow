import { ModelTraceChatMessage } from '../ModelTrace.types';
import { prettyPrintChatMessage } from '../ModelTraceExplorer.utils';

export const normalizeDspyChatInput = (obj: any): ModelTraceChatMessage[] | null => {
  if ('question' in obj) {
    const message = prettyPrintChatMessage({ type: 'message', content: obj.question, role: 'user' });
    return message && [message];
  }

  return null;
};

export const normalizeDspyChatOutput = (obj: any): ModelTraceChatMessage[] | null => {
  if ('answer' in obj) {
    const message = prettyPrintChatMessage({ type: 'message', content: obj.answer, role: 'assistant' });
    return message && [message];
  }

  return null;
};
