import { has } from 'lodash';
import { ModelTraceChatMessage } from '../ModelTrace.types';
import { prettyPrintChatMessage } from '../ModelTraceExplorer.utils';

export const normalizeDspyChatInput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (has(obj, 'question')) {
    const message = prettyPrintChatMessage({ type: 'message', content: obj.question, role: 'user' });
    return message && [message];
  }

  return null;
};

export const normalizeDspyChatOutput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (has(obj, 'answer')) {
    const message = prettyPrintChatMessage({ type: 'message', content: obj.answer, role: 'assistant' });
    return message && [message];
  }

  return null;
};
