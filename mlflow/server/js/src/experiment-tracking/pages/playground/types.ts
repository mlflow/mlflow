export type ChatRole = 'system' | 'user' | 'assistant' | 'tool';

// Registry prompt type, mirroring PROMPT_TYPE_TEXT / PROMPT_TYPE_CHAT in `../prompts/utils`.
export type PromptType = 'text' | 'chat';

export interface ChatMessage {
  role: ChatRole;
  content: string | null;
  tool_calls?: ToolCall[];
  // Links a `tool` role message (a tool result) to the assistant tool call it answers.
  tool_call_id?: string;
}

export interface ToolCall {
  id?: string;
  type?: 'function';
  function?: {
    name?: string;
    arguments?: string;
  };
}

// In-app message type that may carry per-turn usage data on assistant replies.
// Display-only fields are stripped before being sent to the gateway.
export interface ConversationMessage extends ChatMessage {
  usage?: ChatCompletionUsage;
  // Display-only label for `tool` role messages: the name of the tool whose result this is.
  toolName?: string;
  // True when this assistant reply was generated under a JSON / JSON-schema
  // response format, so its content should render as a JSON code block. Captured
  // at generation time so toggling the response-format control afterward does not
  // retroactively reformat past replies. Display-only; stripped before sending.
  contentIsJson?: boolean;
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

export type ToolChoice = 'auto' | 'required';

export interface PlaygroundTool {
  // Stable client-side id for React keys and per-tool updates; not sent to the gateway.
  id: string;
  // Function name (the `function.name` sent to the gateway).
  name: string;
  // Optional human-readable description shown to the model.
  description: string;
  // JSON text for the function's parameters JSON Schema (the `function.parameters` object).
  params: string;
}

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

export interface LogPlaygroundTraceRequest {
  experiment_id: string;
  // The captured input turns (system/user/prior assistant) that produced the response.
  messages: ChatMessage[];
  // The assistant reply being saved. The backend wraps it in an OpenAI chat-completion envelope.
  response?: unknown;
  // Token usage of the reply, so the saved trace shows token counts and cost.
  usage?: ChatCompletionUsage;
  model?: string;
  params?: PlaygroundParams;
  // Tools the run had available (OpenAI wire shape), recorded so the saved trace is reloadable.
  tools?: unknown[];
  tool_choice?: ToolChoice;
  response_format?: ResponseFormat;
  // The trace this playground run was derived from, for comparison attribution.
  source_trace_id?: string;
}

export interface LogPlaygroundTraceResponse {
  trace_id: string;
}
