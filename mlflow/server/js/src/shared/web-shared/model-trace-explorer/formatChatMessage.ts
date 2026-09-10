import { isNil, isObject, isString } from 'lodash';

import type {
  ModelTraceChatMessage,
  ModelTraceContentType,
  ModelTraceInputAudio,
  ModelTraceToolCall,
  RawModelTraceChatMessage,
} from './ModelTrace.types';

export const prettyPrintToolCall = (toolCall: ModelTraceToolCall): ModelTraceToolCall => {
  let args = toolCall.function?.arguments;
  try {
    args = JSON.stringify(JSON.parse(args), null, 2);
  } catch (e) {
    // use original args
  }
  return {
    id: toolCall.id,
    function: {
      arguments: args,
      name: toolCall.function.name,
    },
  };
};

const formatChatContent = (content?: ModelTraceContentType | null): string | undefined | null => {
  if (isNil(content) || isString(content)) {
    return content;
  }

  const contentParts = content
    // eslint-disable-next-line array-callback-return
    .map((part) => {
      switch (part.type) {
        case 'text':
        case 'input_text':
        case 'output_text':
          return part.text;
        case 'image_url':
          const url = part?.image_url?.url;
          return url ? `![](${url})` : '[image]';
        case 'image': {
          // Anthropic format: {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
          const source = (part as any)?.source;
          const imageData = source?.data;
          if (!imageData) return '[image]';
          if (isString(imageData) && imageData.startsWith('mlflow-attachment://')) {
            return `![](${imageData})`;
          }
          const mediaType = source?.media_type;
          return mediaType ? `![](data:${mediaType};base64,${imageData})` : '[image]';
        }
        case 'input_audio':
          // Audio parts are rendered as <audio> elements by the component,
          // so they are excluded from the markdown string
          return undefined;
      }
    })
    .filter((part) => part !== undefined);

  return contentParts.join('\n\n');
};

const extractAudioParts = (content?: ModelTraceContentType | null): ModelTraceInputAudio[] => {
  if (isNil(content) || isString(content)) {
    return [];
  }
  return content
    .filter(
      (part): part is { type: 'input_audio'; input_audio: ModelTraceInputAudio } =>
        part.type === 'input_audio' &&
        isObject((part as any).input_audio) &&
        isString(((part as any).input_audio as any).data) &&
        isString(((part as any).input_audio as any).format),
    )
    .map((part) => part.input_audio);
};

export const prettyPrintChatMessage = (message: RawModelTraceChatMessage): ModelTraceChatMessage | null => {
  // TODO: support rich rendering of reasoning messages
  if (message.type === 'reasoning') {
    return null;
  }

  const audioParts = extractAudioParts(message.content);

  // Extract audio from assistant message output (e.g., gpt-4o-audio-preview response)
  const messageAudio = (message as any).audio;
  if (messageAudio && isString(messageAudio.data)) {
    const format =
      isString(messageAudio.format) && ['wav', 'mp3'].includes(messageAudio.format) ? messageAudio.format : 'wav';
    audioParts.push({ data: messageAudio.data, format });
  }

  return {
    ...message,
    content: formatChatContent(message.content),
    tool_calls: message.tool_calls?.map(prettyPrintToolCall),
    ...(audioParts.length > 0 ? { audioParts } : {}),
  };
};
