/**
 * An ordered piece of an assistant turn. Text and tool calls are kept in arrival
 * order so the transcript can show the work (tool calls) interleaved with the
 * narration, and so tool results/status (filled in later) render where they happened.
 */
export type AssistantPart =
  | { type: 'text'; text: string }
  | {
      type: 'toolCall';
      toolUseId: string;
      name: string;
      input?: Record<string, any>;
      // Filled by follow-up work; kept here so the model doesn't change again.
      status?: 'running' | 'done' | 'error';
      result?: unknown;
    };

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  /**
   * Plain-text mirror of the message. For assistant messages this is the
   * concatenation of the text parts (used for copy and as a fallback when
   * `parts` is absent, e.g. legacy messages).
   */
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  isInterrupted?: boolean;
  /** Ordered parts (text + tool calls) for assistant messages. */
  parts?: AssistantPart[];
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
  currentPage?: string;

  // Sessions
  sessionId?: string;
  selectedSessionIds?: string[];

  // Datasets
  selectedDatasetId?: string;

  // Prompts
  promptName?: string;
  promptVersion?: string;
  comparedPromptVersion?: string;

  // Models
  modelName?: string;
  modelVersion?: string;
  selectedModelVersions?: string[];

  // Scorers/Judges
  selectedScorerName?: string;
}

/** All known context keys */
export type AssistantContextKey = keyof KnownAssistantContext;

/** Cumulative token usage reported by the provider for the current session. */
export interface TokenUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  /**
   * Estimated cumulative cost in USD, or null when no turn could be priced
   * (e.g. local/unknown models absent from the pricing catalog).
   */
  costUsd: number | null;
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
  /** Whether setup is complete (provider selected in config) */
  setupComplete: boolean;
  /** Whether config is being loaded */
  isLoadingConfig: boolean;
  /** Whether the server is running locally (localhost) */
  isLocalServer: boolean;
  /** Cumulative token usage for the session (best-effort; only some providers report it) */
  tokenUsage: TokenUsage;
}

export interface AssistantAgentActions {
  /** Open the Assistant panel */
  openPanel: () => void;
  /** Close the Assistant panel */
  closePanel: () => void;
  /** Send a message to Assistant */
  sendMessage: (message: string) => void;
  /** Regenerate the last assistant response */
  regenerateLastMessage: () => void;
  /** Reset the conversation */
  reset: () => void;
  /** Cancel the current streaming session */
  cancelSession: () => void;
  /** Fetch/refresh config from backend */
  refreshConfig: () => Promise<void>;
  /** Mark setup as complete (after wizard finishes) */
  completeSetup: () => void;
}

export type AssistantAgentContextType = AssistantAgentState & AssistantAgentActions;

/**
 * Request body for sending a message.
 */
export interface MessageRequest {
  session_id?: string;
  message: string;
  experiment_id?: string;
  context?: KnownAssistantContext & Record<string, unknown>;
}

/**
 * Result from the /health endpoint.
 * Status codes: 412 = CLI not installed, 401 = not authenticated, 404 = provider not found
 */
export type HealthCheckResult = { ok: true } | { ok: false; error: string; status: number };

/**
 * Permission settings for the assistant provider.
 */
export interface PermissionsConfig {
  allow_edit_files: boolean;
  allow_read_docs: boolean;
  full_access: boolean;
}

/**
 * Provider configuration.
 */
export interface ProviderConfig {
  model: string;
  selected: boolean;
  permissions: PermissionsConfig;
  base_url?: string;
  api_key?: string;
}

/**
 * Project configuration (experiment to workspace mapping).
 */
export interface ProjectConfig {
  type: 'local';
  location: string;
}

/**
 * Full assistant configuration from /config endpoint.
 */
export interface AssistantConfig {
  providers: Record<string, ProviderConfig>;
  projects: Record<string, ProjectConfig>;
  skills_location?: string;
}

/**
 * Config update request - allows null to remove entries.
 */
export interface AssistantConfigUpdate {
  providers?: Record<string, Partial<ProviderConfig>>;
  projects?: Record<string, ProjectConfig | null>;
}

/**
 * Setup wizard step type.
 */
export type SetupStep = 'provider' | 'connection' | 'project' | 'complete';

/**
 * Response from installing skills.
 */
export interface InstallSkillsResponse {
  installed_skills: string[];
  skills_directory: string;
}

/**
 * Authentication state for provider connection check.
 */
export type AuthState = 'checking' | 'cli_not_installed' | 'not_authenticated' | 'authenticated';
