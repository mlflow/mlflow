export type ChatRole = 'system' | 'user' | 'assistant';

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export type ResponseFormatType = 'text' | 'json_object' | 'json_schema';

export type ResponseFormat = { type: 'json_object' } | { type: 'json_schema'; json_schema: unknown };

export type ToolChoice = 'auto' | 'none' | 'required';

export interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  top_k?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  stop?: string[];
  tools?: unknown[];
  tool_choice?: ToolChoice;
  response_format?: ResponseFormat;
}

export interface PlaygroundParams {
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  top_k?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  stop?: string[];
}

export interface ChatCompletionUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
}

export interface ChatCompletionChoice {
  index?: number;
  message?: ChatMessage;
  finish_reason?: string;
}

export interface ChatCompletionResponse {
  id?: string;
  model?: string;
  choices: ChatCompletionChoice[];
  usage?: ChatCompletionUsage;
}
