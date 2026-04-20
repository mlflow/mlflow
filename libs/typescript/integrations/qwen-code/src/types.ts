/**
 * Types for Qwen Code transcript parsing.
 *
 * Qwen Code transcripts are JSONL files with ChatRecords linked by
 * uuid/parentUuid but emitted in chronological order. Messages use a
 * Gemini-style `{role, parts: [...]}` envelope, and parts can be one of:
 *   - `{text, thought?}` — model/user text (thought=true marks internal reasoning)
 *   - `{functionCall: {id, name, args}}` — model requesting a tool call
 *   - `{functionResponse: {id, name, response}}` — result returned to the model
 *
 * Tool results additionally appear as standalone records with
 * `type: 'tool_result'` that carry a `toolCallResult` block with
 * `{callId, status, resultDisplay}` — matched to the assistant's
 * functionCall by call id.
 *
 * Location: ~/.qwen/projects/<project-id>/chats/<sessionId>.jsonl
 */

/**
 * A single ChatRecord in the Qwen Code JSONL transcript.
 */
export interface ChatRecord {
  uuid: string;
  parentUuid: string | null;
  sessionId: string;
  timestamp: string;
  type: 'user' | 'assistant' | 'system' | 'tool_result';
  /** Gemini-style message envelope; `system` records may have no parts. */
  message?: GeminiMessage | string;
  model?: string;
  usageMetadata?: UsageMetadata;
  /** Present on `tool_result` records. Matched to an assistant functionCall by callId. */
  toolCallResult?: ToolCallResult;
  cwd?: string;
  gitBranch?: string;
  version?: string;
  contextWindowSize?: number;
  subtype?: string;
  systemPayload?: unknown;
}

/**
 * Gemini-style message envelope used by Qwen Code.
 */
export interface GeminiMessage {
  role: string;
  parts: GeminiPart[];
}

/**
 * One piece of a Gemini message. Real transcripts contain four shapes:
 * text with optional `thought` flag, function calls, and function responses.
 */
export type GeminiPart = TextPart | FunctionCallPart | FunctionResponsePart;

export interface TextPart {
  text: string;
  /** Internal chain-of-thought reasoning. Exclude from user-facing content. */
  thought?: boolean;
}

export interface FunctionCallPart {
  functionCall: FunctionCall;
}

export interface FunctionResponsePart {
  functionResponse: FunctionResponse;
}

export interface FunctionCall {
  id: string;
  name: string;
  args?: Record<string, unknown>;
}

export interface FunctionResponse {
  id: string;
  name: string;
  response?: Record<string, unknown>;
}

/**
 * Token usage metadata on assistant records. Qwen uses Gemini-style keys
 * (promptTokenCount / candidatesTokenCount / totalTokenCount), with
 * OpenAI-style fallbacks occasionally appearing.
 */
export interface UsageMetadata {
  promptTokenCount?: number;
  candidatesTokenCount?: number;
  totalTokenCount?: number;
  thoughtsTokenCount?: number;
  cachedContentTokenCount?: number;
  input_tokens?: number;
  output_tokens?: number;
}

/**
 * Tool result payload on `tool_result` records.
 *
 * Status values observed in real transcripts:
 *   - `success` — tool completed normally
 *   - `cancelled` — user declined a permission prompt
 *   - (other values are treated as failures defensively)
 */
export interface ToolCallResult {
  callId: string;
  status: string;
  /** Can be a plain string or a structured object (e.g. file diff for write_file). */
  resultDisplay?: string | Record<string, unknown>;
}

/**
 * Stop hook input received via stdin.
 */
export interface StopHookInput {
  session_id: string;
  transcript_path: string | null;
  cwd: string;
  hook_event_name: string;
  timestamp: string;
}

/**
 * OpenAI chat-format tool call, attached to assistant messages in LLM span
 * inputs so the MLflow Chat view renders tool invocations correctly.
 */
export interface ToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

/**
 * OpenAI chat-format message used in LLM span inputs.
 */
export interface ChatMessage {
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string | null;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}
