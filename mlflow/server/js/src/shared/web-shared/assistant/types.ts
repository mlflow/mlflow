export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
}

/**
 * Information about a tool being used by the assistant.
 */
export interface ToolUseInfo {
  id: string;
  name: string;
  description?: string;
  input?: Record<string, any>;
}

/**
 * Known context keys for the assistant.
 * Type-safe registration for common context values.
 */
export interface KnownAssistantContext {
  experimentId?: string;
  traceId?: string;
  selectedTraceIds?: string[];
  runId?: string;
  selectedRunIds?: string[];
}

export interface AssistantAgentState {
  /** Whether the Assistant panel is open */
  isPanelOpen: boolean;
  /** Session ID for conversation continuity */
  sessionId: string | null;
  /** Chat message history */
  messages: ChatMessage[];
  /** Whether a response is being streamed */
  isStreaming: boolean;
  /** Error message if any */
  error: string | null;
  /** Current tool usage status (e.g., "Reading file...", "Searching...") */
  currentStatus: string | null;
  /** Active tools being used by the assistant */
  activeTools: ToolUseInfo[];
}

export interface AssistantAgentActions {
  /** Open the Assistant panel */
  openPanel: () => void;
  /** Close the Assistant panel */
  closePanel: () => void;
  /** Send a message to Assistant */
  sendMessage: (message: string) => void;
  /** Reset the conversation */
  reset: () => void;
}

export type AssistantAgentContextType = AssistantAgentState & AssistantAgentActions;

/**
 * Request body for sending a message.
 */
export interface MessageRequest {
  session_id?: string;
  message: string;
  /** Experiment ID for linking to MLflow experiment */
  experiment_id?: string;
  /** Page-specific context (traceId, selectedTraceIds, etc.) */
  context?: KnownAssistantContext & Record<string, unknown>;
}
