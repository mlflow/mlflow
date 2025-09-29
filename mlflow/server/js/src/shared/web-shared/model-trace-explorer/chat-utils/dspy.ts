import { has, isArray, isString } from 'lodash';

import type { ModelTraceChatMessage } from '../ModelTrace.types';
import { prettyPrintChatMessage } from '../ModelTraceExplorer.utils';

export const normalizeDspyChatInput = (obj: unknown): ModelTraceChatMessage[] | null => {
  // Handle DSPy format with messages array
  if (has(obj, 'messages') && isArray((obj as any).messages)) {
    const messages = (obj as any).messages;
    return messages
      .map((msg: any) => prettyPrintChatMessage({ type: 'message', content: msg.content, role: msg.role }))
      .filter(Boolean);
  }

  return null;
};

export const normalizeDspyChatOutput = (obj: unknown): ModelTraceChatMessage[] | null => {
  // Handle DSPy format with array output
  if (isArray(obj) && obj.length > 0 && obj.every(isString)) {
    // Join all output strings into one assistant message
    const content = obj.join('\n');
    const message = prettyPrintChatMessage({ type: 'message', content, role: 'assistant' });
    return message && [message];
  }

  return null;
};
