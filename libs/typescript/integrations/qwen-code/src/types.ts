/**
 * Types for Qwen Code transcript parsing.
 *
 * Qwen Code transcripts are JSONL files with tree-structured ChatRecords.
 * Each record has uuid/parentUuid for parent-child relationships.
 * Messages use Gemini-style format: {role, parts: [{text}]}.
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
  type: 'user' | 'assistant' | 'system';
  /** Gemini-style message: {role, parts: [{text}]} or plain string */
  message: GeminiMessage | string;
  model?: string;
  usageMetadata?: UsageMetadata;
  toolCallResult?: ToolCallResult;
  cwd?: string;
  gitBranch?: string;
}

/**
 * Gemini-style message format used by Qwen Code.
 */
export interface GeminiMessage {
  role: string;
  parts: GeminiPart[];
}

export interface GeminiPart {
  text: string;
  thought?: boolean;
}

/**
 * Token usage metadata from Qwen's usageMetadata field.
 */
export interface UsageMetadata {
  promptTokenCount?: number;
  candidatesTokenCount?: number;
  totalTokenCount?: number;
  // Fallback keys (OpenAI-style)
  input_tokens?: number;
  output_tokens?: number;
}

/**
 * Tool call result embedded in a ChatRecord.
 */
export interface ToolCallResult {
  name: string;
  input?: Record<string, unknown>;
  output?: string;
}

/**
 * Stop hook input received via stdin.
 * Qwen Code hooks receive JSON on stdin with session_id and transcript_path.
 */
export interface StopHookInput {
  session_id: string;
  transcript_path: string | null;
  cwd: string;
  hook_event_name: string;
  timestamp: string;
}
