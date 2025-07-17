import type { RawModelTraceChatMessage } from '../ModelTrace.types';

export type OpenAIResponsesFunctionCallOutput = {
  type: 'function_call_output';
  call_id: string;
  output: string;
};

export type OpenAIResponsesFunctionCall = {
  type: 'function_call';
  call_id: string;
  name: string;
  arguments: string;
  id: string;
};

export type OpenAIResponsesOutputItem =
  | RawModelTraceChatMessage
  | OpenAIResponsesFunctionCall
  | OpenAIResponsesFunctionCallOutput;

export type OpenAIResponsesStreamingOutputChunk = {
  type: 'response.output_item.done';
  role: 'assistant';
  item: OpenAIResponsesOutputItem;
};
