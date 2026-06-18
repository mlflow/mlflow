export type ChatRole = 'system' | 'user' | 'assistant';

// Registry prompt type, mirroring PROMPT_TYPE_TEXT / PROMPT_TYPE_CHAT in `../prompts/utils`.
export type PromptType = 'text' | 'chat';

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

// In-app message type that may carry per-turn usage data on assistant replies.
// Stripped to `{role, content}` before being sent to the gateway.
export interface ConversationMessage extends ChatMessage {
  usage?: ChatCompletionUsage;
}

export type ResponseFormatType = 'text' | 'json_object' | 'json_schema';

export type ResponseFormat =
  | { type: 'json_object' }
  | {
      type: 'json_schema';
      // OpenAI-compatible envelope: gateway providers (Gemini, Anthropic)
      // expect json_schema as { name, schema, strict }, not the raw schema.
      json_schema: { name: string; schema: unknown; strict?: boolean };
    };

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
