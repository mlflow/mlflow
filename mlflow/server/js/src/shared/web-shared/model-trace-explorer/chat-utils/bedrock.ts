import { compact, has, isArray, isNil, isObject, isString } from 'lodash';

import type { ModelTraceChatMessage, ModelTraceContentParts } from '../ModelTrace.types';
import { prettyPrintToolCall } from '../ModelTraceExplorer.utils';

interface BedrockToolUse {
  toolUseId: string;
  name: string;
  input: string | Record<string, any>;
}

interface BedrockToolResult {
  toolUseId: string;
  content: BedrockContentBlock[];
}

interface BedrockContentBlock {
  text?: string;
  json?: any;
  image?: {
    source: {
      bytes: string | Uint8Array;
    };
    format: string;
  };
  toolUse?: BedrockToolUse;
  toolResult?: BedrockToolResult;
}

interface BedrockMessage {
  role: 'user' | 'assistant' | 'system';
  content: BedrockContentBlock[];
}

const parseBedrockContent = (content: BedrockContentBlock): ModelTraceContentParts | null => {
  if (content.text) {
    return { type: 'text', text: content.text };
  }

  if (content.json) {
    return { type: 'text', text: JSON.stringify(content.json) };
  }

  if (content.image) {
    const bytes = content.image.source.bytes;
    const format = content.image.format;
    let data: string;

    if (typeof bytes === 'string') {
      data = bytes;
    } else {
      // Convert Uint8Array to base64 string
      const buffer = Buffer.from(bytes);
      data = buffer.toString('base64');
    }

    const imageFormat = `image/${format}`;
    return {
      type: 'image_url',
      image_url: { url: `data:${imageFormat};base64,${data}` },
    };
  }

  // Unsupported content types (video, document) are ignored
  return null;
};

const convertBedrockMessageToChatMessage = (message: BedrockMessage): ModelTraceChatMessage => {
  let role: 'user' | 'assistant' | 'system' | 'tool' = message.role;
  const contents: ModelTraceContentParts[] = [];
  const toolCalls: any[] = [];
  let toolCallId: string | undefined;

  for (const content of message.content) {
    if (content.toolUse) {
      const toolCall = content.toolUse;
      const input = typeof toolCall.input === 'string' ? toolCall.input : JSON.stringify(toolCall.input);

      toolCalls.push(
        prettyPrintToolCall({
          id: toolCall.toolUseId,
          function: {
            name: toolCall.name,
            arguments: input,
          },
        }),
      );
    } else if (content.toolResult) {
      toolCallId = content.toolResult.toolUseId;
      role = 'tool';

      for (const resultContent of content.toolResult.content) {
        const parsedContent = parseBedrockContent(resultContent);
        if (parsedContent) {
          contents.push(parsedContent);
        }
      }
    } else {
      const parsedContent = parseBedrockContent(content);
      if (parsedContent) {
        contents.push(parsedContent);
      }
    }
  }

  // Convert content parts to string
  let contentString = '';
  if (contents.length > 0) {
    contentString = contents
      .map((part) => {
        if (part.type === 'text') {
          return part.text;
        } else if (part.type === 'image_url') {
          return `[Image: ${part.image_url.url}]`;
        }
        return '';
      })
      .filter((text) => text.length > 0)
      .join(' ');
  }

  const chatMessage: ModelTraceChatMessage = {
    role: role,
    content: contentString,
  };

  if (toolCalls.length > 0) {
    chatMessage.tool_calls = toolCalls;
  }

  if (toolCallId) {
    chatMessage.tool_call_id = toolCallId;
  }

  return chatMessage;
};

const isBedrockMessage = (obj: unknown): obj is BedrockMessage => {
  if (!isObject(obj)) {
    return false;
  }

  const hasRole = has(obj, 'role') && isString(obj.role) && ['user', 'assistant', 'system'].includes(obj.role);
  const hasContent = has(obj, 'content') && isArray(obj.content);

  return hasRole && hasContent;
};

export const normalizeBedrockChatInput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (isNil(obj)) {
    return null;
  }

  // Handle case where input has 'messages' key
  if (isObject(obj) && 'messages' in obj) {
    const messages = (obj as any).messages;
    if (isArray(messages) && messages.length > 0 && messages.every(isBedrockMessage)) {
      return compact(messages.map(convertBedrockMessageToChatMessage));
    }
  }

  return null;
};

export const normalizeBedrockChatOutput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (isNil(obj)) {
    return null;
  }

  // Handle case where output has 'output' -> 'message' structure
  if (isObject(obj) && 'output' in obj) {
    const output = (obj as any).output;
    if (isObject(output) && 'message' in output) {
      const message = output.message;
      if (isBedrockMessage(message)) {
        return [convertBedrockMessageToChatMessage(message)];
      }
    }
  }

  return null;
};
