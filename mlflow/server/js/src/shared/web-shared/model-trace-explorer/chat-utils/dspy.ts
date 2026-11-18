import { has, isArray, isEmpty, isObject, isString } from 'lodash';

import type { ModelTraceChatMessage, ModelTraceContentParts, ModelTraceToolCall } from '../ModelTrace.types';
import { prettyPrintChatMessage } from '../ModelTraceExplorer.utils';

// DSPy 3.5+ calls LMs with lm15 `Request`/`Response` objects. Their messages hold a list of
// typed `parts` (text, image, tool_call, tool_result, thinking, ...) instead of a `content` string.
type DspyLm15Message = {
  role: 'user' | 'assistant' | 'tool' | 'developer';
  parts: unknown[];
};

type DspyLm15Request = {
  messages: DspyLm15Message[];
  system?: unknown;
};

const DSPY_LM15_ROLES = ['user', 'assistant', 'tool', 'developer'];

const isDspyLm15Message = (obj: unknown): obj is DspyLm15Message =>
  isObject(obj) && DSPY_LM15_ROLES.includes((obj as any).role) && isArray((obj as any).parts);

const isDspyLm15Request = (obj: unknown): obj is DspyLm15Request =>
  isObject(obj) && isArray((obj as any).messages) && (obj as any).messages.every(isDspyLm15Message);

// Convert a snake_case or space-separated variable name into Title Case.
// e.g. "tool_name_0" → "Tool Name 0"
const titleCase = (name: string) => name.replace(/_/g, ' ').replace(/\b[a-z]/g, (ch) => ch.toUpperCase());

// Convert standalone `[[ ## name ## ]]` markers into markdown headings.
// - Markers that sit on their own line become `#### Title Cased Name`
// - The `[[ ## completed ## ]]` terminator is removed entirely
// - Inline references (e.g. inside backticks mid-sentence) are left unchanged
export const formatDspySections = (text: string): string =>
  text.replace(/^[ \t]*\[\[\s*##\s*(.*?)\s*##\s*\]\][ \t]*$/gm, (_match, name: string) => {
    if (name.toLowerCase() === 'completed') {
      return '';
    }
    return `#### ${titleCase(name.trim())}`;
  });

const formatDspyText = (text: string) => toMarkdownWithHardBreaks(formatDspySections(text));

// If the output is valid JSON (e.g. from JSONAdapter), render it as a formatted code block
const formatDspyOutputText = (text: string) => {
  try {
    const parsed = JSON.parse(text);
    if (typeof parsed === 'object' && parsed !== null) {
      return '```json\n' + JSON.stringify(parsed, null, 2) + '\n```';
    }
  } catch {
    // Not JSON, render as text
  }
  return formatDspyText(text);
};

const getDspyLm15ImageUrl = (part: any): string | undefined => {
  if (isString(part.url)) {
    return part.url;
  }
  if (isString(part.data) && isString(part.media_type)) {
    // Inline images may already be extracted into mlflow-attachment:// URIs
    return part.data.startsWith('mlflow-attachment://') ? part.data : `data:${part.media_type};base64,${part.data}`;
  }
  return undefined;
};

// Part types that `normalizeDspyLm15Message` renders outside the message content
const DSPY_LM15_NON_CONTENT_PART_TYPES = ['tool_call', 'tool_result', 'thinking'];

const toDspyLm15ContentParts = (parts: unknown[], formatText: (text: string) => string): ModelTraceContentParts[] =>
  parts.flatMap((part: any): ModelTraceContentParts[] => {
    if (part?.type === 'text' || part?.type === 'refusal') {
      return isString(part.text) ? [{ type: 'text', text: formatText(part.text) }] : [];
    }
    if (part?.type === 'image') {
      const url = getDspyLm15ImageUrl(part);
      return [url ? { type: 'image_url', image_url: { url } } : { type: 'text', text: '[image]' }];
    }
    // Show a placeholder for parts the chat view can't render, e.g. audio, document or citation
    if (isString(part?.type) && !DSPY_LM15_NON_CONTENT_PART_TYPES.includes(part.type)) {
      return [{ type: 'text', text: `[${part.type}]` }];
    }
    return [];
  });

const normalizeDspyLm15Message = (message: DspyLm15Message): ModelTraceChatMessage[] => {
  const messages: ModelTraceChatMessage[] = [];
  const toolCalls: ModelTraceToolCall[] = [];
  const thinkingParts: string[] = [];

  for (const part of message.parts as any[]) {
    if (part?.type === 'tool_result' && isArray(part.content)) {
      const toolMessage = prettyPrintChatMessage({
        type: 'message',
        role: 'tool',
        tool_call_id: part.id,
        name: isString(part.name) ? part.name : undefined,
        // The chat view shows tool responses as plain text, or as a code block when the content
        // parses as JSON, so keep the text unformatted
        content: toDspyLm15ContentParts(part.content, (text) => text),
      });
      if (toolMessage) {
        messages.push(toolMessage);
      }
    } else if (part?.type === 'tool_call' && isString(part.name)) {
      toolCalls.push({
        id: isString(part.id) ? part.id : part.name,
        function: { name: part.name, arguments: JSON.stringify(part.input ?? {}) },
      });
    } else if (part?.type === 'thinking' && isString(part.text)) {
      thinkingParts.push(part.text);
    }
  }

  const content = toDspyLm15ContentParts(
    message.parts,
    message.role === 'assistant' ? formatDspyOutputText : formatDspyText,
  );
  if (content.length > 0 || toolCalls.length > 0 || thinkingParts.length > 0) {
    const chatMessage = prettyPrintChatMessage({
      type: 'message',
      role: message.role,
      content: content.length > 0 ? content : undefined,
      tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
    });
    if (chatMessage) {
      messages.push(thinkingParts.length > 0 ? { ...chatMessage, reasoning: thinkingParts.join('\n\n') } : chatMessage);
    }
  }

  return messages;
};

const normalizeDspyLm15Request = (request: DspyLm15Request): ModelTraceChatMessage[] => {
  const messages: ModelTraceChatMessage[] = [];
  const { system } = request;
  if (!isEmpty(system) && (isString(system) || isArray(system))) {
    const systemMessage = prettyPrintChatMessage({
      type: 'message',
      role: 'system',
      content: isString(system) ? formatDspyText(system) : toDspyLm15ContentParts(system, formatDspyText),
    });
    if (systemMessage) {
      messages.push(systemMessage);
    }
  }
  return [...messages, ...request.messages.flatMap(normalizeDspyLm15Message)];
};

export const normalizeDspyChatInput = (obj: unknown): ModelTraceChatMessage[] | null => {
  // Handle DSPy format with messages array
  if (has(obj, 'messages') && isArray((obj as any).messages)) {
    const messages = (obj as any).messages;
    return messages
      .map((msg: any) =>
        prettyPrintChatMessage({
          type: 'message',
          content: isString(msg.content) ? formatDspyText(msg.content) : msg.content,
          role: msg.role,
        }),
      )
      .filter(Boolean);
  }

  // DSPy 3.5+ calls the LM with an lm15 `Request`, or a string that becomes one user message
  const prompt = has(obj, 'prompt') ? (obj as any).prompt : undefined;
  if (isDspyLm15Request(prompt)) {
    const messages = normalizeDspyLm15Request(prompt);
    return messages.length > 0 ? messages : null;
  }
  if (isString(prompt)) {
    const message = prettyPrintChatMessage({ type: 'message', content: formatDspyText(prompt), role: 'user' });
    return message && [message];
  }

  return null;
};

export const normalizeDspyChatOutput = (obj: unknown): ModelTraceChatMessage[] | null => {
  // Handle DSPy format with array output
  if (isArray(obj) && obj.length > 0 && obj.every(isString)) {
    const content = formatDspyOutputText(obj.join('\n'));
    const message = prettyPrintChatMessage({ type: 'message', content, role: 'assistant' });
    return message && [message];
  }

  // DSPy 3.5+ returns an lm15 `Response` that holds one assistant message
  if (has(obj, 'message') && isDspyLm15Message((obj as any).message)) {
    const messages = normalizeDspyLm15Message((obj as any).message);
    return messages.length > 0 ? messages : null;
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
