/**
 * Types for Codex CLI notify hook integration.
 *
 * Codex CLI fires a `notify` hook after each agent turn, passing a JSON
 * argument with the turn data. This is configured in ~/.codex/config.toml:
 *   notify = ["node", "/path/to/stop.js"]
 *
 * The JSON is passed as the first command-line argument (argv[2]).
 *
 * Reference: https://developers.openai.com/codex/hooks
 */

/**
 * Notify hook payload — passed as a CLI argument JSON string.
 * Fired after each agent turn completes.
 */
export interface NotifyPayload {
  type: 'agent-turn-complete';
  'thread-id': string;
  'turn-id': string;
  cwd: string;
  client: string;
  'input-messages': string[];
  'last-assistant-message': string;
}

/**
 * Types below are for transcript parsing (rollout JSONL files).
 * Defined in codex-rs/protocol/src/protocol.rs (tagged enum `RolloutItem`).
 * Stored at ~/.codex/sessions/YYYY/MM/DD/rollout-<timestamp>-<session_id>.jsonl.
 */

/**
 * A single line in the Codex rollout JSONL transcript.
 */
export interface RolloutLine {
  timestamp: string;
  type: 'session_meta' | 'response_item' | 'event_msg' | 'turn_context' | 'compacted';
  payload: SessionMetaPayload | ResponseItemPayload | EventMsgPayload | Record<string, unknown>;
}

export interface SessionMetaPayload {
  id: string;
  timestamp: string;
  cwd: string;
  originator: string;
  cli_version: string;
  source: string;
  model_provider?: string;
}

export interface ResponseItemPayload {
  type: 'message' | 'function_call' | 'function_call_output' | 'reasoning';
  role?: 'user' | 'assistant' | 'developer';
  content?: ContentBlock[];
  name?: string;
  call_id?: string;
  arguments?: string;
  output?: string;
}

export interface ContentBlock {
  type: 'input_text' | 'output_text';
  text: string;
}

export interface EventMsgPayload {
  type: string;
  info?: TokenCountInfo;
}

export interface TokenCountInfo {
  last_token_usage?: TokenUsage;
  total_token_usage?: TokenUsage;
}

export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cached_input_tokens?: number;
  reasoning_output_tokens?: number;
}

/**
 * OpenAI chat-format tool call, used on assistant messages.
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
 * OpenAI chat-format message used in LLM span inputs. Matches the message
 * structure the MLflow UI Chat view renders.
 */
export interface ChatMessage {
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string | null;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}
