/**
 * TypeScript interfaces for Claude Code transcript entries.
 */

// ============================================================================
// Content block types
// ============================================================================

export interface TextBlock {
  type: 'text';
  text: string;
}

export interface ThinkingBlock {
  type: 'thinking';
  thinking: string;
}

export interface ToolUseBlock {
  type: 'tool_use';
  id: string;
  name: string;
  input: Record<string, unknown>;
}

export interface ToolResultBlock {
  type: 'tool_result';
  tool_use_id: string;
  content: string;
  is_error?: boolean;
  toolUseResult?: {
    status?: string;
    agentId?: string;
    totalDurationMs?: number;
  };
}

export type ContentBlock = TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock;

// ============================================================================
// Token usage
// ============================================================================

export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
}

// ============================================================================
// Message content
// ============================================================================

export interface MessageContent {
  role: 'user' | 'assistant';
  content: string | ContentBlock[];
  id?: string;
  model?: string;
  usage?: TokenUsage;
}

// ============================================================================
// Transcript entries
// ============================================================================

export interface TranscriptEntry {
  type: 'user' | 'assistant' | 'progress' | 'queue-operation';
  message?: MessageContent;
  timestamp?: string | number;
  version?: string;
  permissionMode?: string;
  toolUseResult?: ToolUseResultInfo;
  sessionId?: string;
  parentToolUseID?: string;
  data?: ProgressData;
  operation?: string;
  content?: string;
  agentId?: string;
  isCompactSummary?: boolean;
}

export interface ToolUseResultInfo {
  success?: boolean;
  commandName?: string;
  agentId?: string;
  status?: string;
  totalDurationMs?: number;
}

export interface ProgressData {
  type?: string;
  agentId?: string;
  prompt?: string;
  message?: TranscriptEntry;
}

// ============================================================================
// Hook input/output
// ============================================================================

export interface StopHookInput {
  session_id: string;
  transcript_path: string;
}

// ============================================================================
// Internal types
// ============================================================================

export interface ToolResultInfo {
  content: string;
  isError: boolean;
  agentId?: string;
}

export interface SubagentGroup {
  prompt: string;
  messages: TranscriptEntry[];
  timestamp?: string | number;
}
