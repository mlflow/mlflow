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

export type OpenAIResponsesImageGenerationCall = {
  type: 'image_generation_call';
  id: string;
  result: string | null;
  status: string;
  output_format: 'png' | 'jpeg' | 'webp';
};

export type OpenAIResponsesInputText = {
  text: string;
  type: 'input_text';
};

export type OpenAIResponsesInputImage = {
  detail: 'high' | 'low' | 'auto';
  type: 'input_image';
  file_id?: string;
  image_url?: string;
};

export type OpenAIResponsesInputFile = {
  type: 'input_file';
  file_data?: string;
  file_id?: string;
  file_url?: string;
  filename?: string;
};

export type OpenAIResponsesInputMessageRole = 'user' | 'assistant' | 'system' | 'developer';

export type OpenAIResponsesInputMessage = {
  content: string | (OpenAIResponsesInputText | OpenAIResponsesInputImage | OpenAIResponsesInputFile)[];
  role: OpenAIResponsesInputMessageRole;
};

// type OpenAIResponsesItem =
//   | InputMessage
//   | OutputMessage
//   | FileSearchToolCall
//   | ComputerToolCall
//   | ComputerToolCallOutput
//   | WebSearchToolCall
//   | FunctionToolCall
//   | FunctionToolCallOutput
//   | Reasoning
//   | ImageGenerationToolCall
//   | CodeInterpreterToolCall
//   | LocalShellCall
//   | LocalShellCallOutput
//   | MCPListTools
//   | MCPApprovalRequest
//   | MCPApprovalResponse
//   | MCPToolCall;

// type OpenAIResponsesItemReference = {
//   id: string;
//   type?: 'item_reference';
// };

// NOTE: these types not supported yet
export type OpenAIResponsesInput =
  | string
  | OpenAIResponsesInputMessage /* | OpenAIResponsesItem | OpenAIResponsesItemReference */[];

export type OpenAIResponsesOutputItem =
  | RawModelTraceChatMessage
  | OpenAIResponsesFunctionCall
  | OpenAIResponsesImageGenerationCall
  | OpenAIResponsesFunctionCallOutput;

export type OpenAIResponsesStreamingOutputChunk = {
  type: 'response.output_item.done';
  role: 'assistant';
  item: OpenAIResponsesOutputItem;
};
