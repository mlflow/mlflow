import { compact, isArray, isObject, isString } from 'lodash';

import type { ModelTraceChatMessage } from '../ModelTrace.types';
import { prettyPrintChatMessage, prettyPrintToolCall } from '../ModelTraceExplorer.utils';

/**
 * Mistral chat normalizer that handles both regular Mistral and Magistral (reasoning) models.
 *
 * Regular Mistral: content is a string
 * Magistral: content is an array of ThinkingChunk and TextChunk objects
 */

/**
 * Extract text from a thinking array: [{ text: '...', type: 'text' }]
 */
const extractThinkingText = (thinkingArray: unknown): string[] => {
  if (!isArray(thinkingArray)) return [];

  const texts: string[] = [];
  for (const item of thinkingArray as Record<string, unknown>[]) {
    const text = item['text'];
    if (isString(text)) {
      texts.push(text);
    }
  }
  return texts;
};

/**
 * Process Mistral content which can be a string or an array of chunks.
 * Magistral reasoning models return content as an array with thinking and text chunks.
 */
const processMistralContent = (
  content: unknown,
): {
  textContent: string | null;
  reasoning: string | null;
} => {
  // Simple string content (regular Mistral)
  if (isString(content)) {
    return { textContent: content, reasoning: null };
  }

  // Array content (Magistral reasoning models)
  if (!isArray(content)) {
    return { textContent: null, reasoning: null };
  }

  const textParts: string[] = [];
  const thinkingParts: string[] = [];

  for (const chunk of content as Record<string, unknown>[]) {
    if (!isObject(chunk)) continue;

    const chunkType = chunk['type'];

    // Thinking chunk: { type: 'thinking', thinking: [{ text: '...', type: 'text' }] }
    if (chunkType === 'thinking') {
      thinkingParts.push(...extractThinkingText(chunk['thinking']));
    }

    // Text chunk: { type: 'text', text: '...' }
    if (chunkType === 'text') {
      const text = chunk['text'];
      if (isString(text)) {
        textParts.push(text);
      }
    }
  }

  return {
    textContent: textParts.length > 0 ? textParts.join('\n\n') : null,
    reasoning: thinkingParts.length > 0 ? thinkingParts.join('\n\n') : null,
  };
};

const normalizeMistralMessage = (message: unknown): ModelTraceChatMessage | null => {
  if (!isObject(message)) return null;

  const msg = message as Record<string, unknown>;
  const role = msg['role'];
  if (!isString(role)) return null;

  const content = msg['content'];
  const { textContent, reasoning } = processMistralContent(content);

  // Handle tool calls if present
  const toolCalls = msg['tool_calls'];
  if (isArray(toolCalls) && toolCalls.length > 0) {
    const validToolCalls = (toolCalls as Record<string, unknown>[]).filter(
      (tc) => isObject(tc) && 'id' in tc && 'function' in tc,
    );
    if (validToolCalls.length > 0) {
      return {
        role: role as 'assistant',
        content: textContent ?? undefined,
        tool_calls: validToolCalls.map((tc) =>
          prettyPrintToolCall({
            id: tc['id'] as string,
            function: tc['function'] as { name: string; arguments: string },
          }),
        ),
        ...(reasoning && { reasoning }),
      };
    }
  }

  // Handle tool results
  const toolCallId = msg['tool_call_id'];
  if (role === 'tool' && isString(toolCallId)) {
    return {
      role: 'tool',
      tool_call_id: toolCallId,
      content: textContent ?? '',
    };
  }

  // Handle regular messages
  const chatMessage = prettyPrintChatMessage({
    type: 'message',
    content: textContent ?? '',
    role: role as 'user' | 'assistant' | 'system',
  });

  if (chatMessage && reasoning) {
    return { ...chatMessage, reasoning };
  }

  return chatMessage;
};

export const normalizeMistralChatInput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (!isObject(obj)) return null;

  const objRecord = obj as Record<string, unknown>;

  // Check for messages array
  const messages = objRecord['messages'];
  if (!isArray(messages) || messages.length === 0) return null;

  // Try to normalize each message
  const normalizedMessages = compact(messages.map(normalizeMistralMessage));

  // Return if we got at least one valid message
  return normalizedMessages.length > 0 ? normalizedMessages : null;
};

export const normalizeMistralChatOutput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (!isObject(obj)) return null;

  const objRecord = obj as Record<string, unknown>;

  // Check for choices array (chat completion format)
  const choices = objRecord['choices'];
  if (!isArray(choices) || choices.length === 0) return null;

  // Try to normalize messages from choices
  const normalizedMessages = compact(
    (choices as Record<string, unknown>[]).map((choice) => {
      if (!isObject(choice)) return null;
      const message = choice['message'];
      return normalizeMistralMessage(message);
    }),
  );

  // Return if we got at least one valid message
  return normalizedMessages.length > 0 ? normalizedMessages : null;
};
