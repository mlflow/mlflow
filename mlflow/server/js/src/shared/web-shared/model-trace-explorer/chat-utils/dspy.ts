import { has, isArray, isString } from 'lodash';

import type { ModelTraceChatMessage } from '../ModelTrace.types';
import { prettyPrintChatMessage } from '../ModelTraceExplorer.utils';

export const normalizeDspyChatInput = (obj: unknown): ModelTraceChatMessage[] | null => {
  // Handle DSPy format with messages array
  if (has(obj, 'messages') && isArray((obj as any).messages)) {
    const messages = (obj as any).messages;
    return messages
      .map((msg: any) =>
        prettyPrintChatMessage({
          type: 'message',
          content: isString(msg.content) ? toMarkdownWithHardBreaks(msg.content) : msg.content,
          role: msg.role,
        }),
      )
      .filter(Boolean);
  }

  return null;
};

export const normalizeDspyChatOutput = (obj: unknown): ModelTraceChatMessage[] | null => {
  // Handle DSPy format with array output
  if (isArray(obj) && obj.length > 0 && obj.every(isString)) {
    // Join all output strings into one assistant message
    const content = toMarkdownWithHardBreaks(obj.join('\n'));
    const message = prettyPrintChatMessage({ type: 'message', content, role: 'assistant' });
    return message && [message];
  }

  return null;
};

// Markdown treats single newlines as spaces. For DSPy prompts that are plain text
// we convert single newlines into hard line breaks while preserving paragraphs.
// Only the single line break will be updated, for example,
// - "foo\nbar" -> "foo  \nbar" (two spaces inserted before \n)
// - "foo\n\nbar" -> "foo\n\nbar" (no change)
const toMarkdownWithHardBreaks = (text: string) =>
  text.replace(/\r\n/g, '\n').replace(/(^|[^\n])\n(?!\n)/g, (_m, p1) => `${p1}  \n`);
