export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  isInterrupted?: boolean;
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
 * A pending tool-call permission request surfaced to the user for a Yes/No
 * decision when the session is not in full-access mode.
 */
export interface PermissionRequest {
  /**
   * The session that produced this request, so a decision targets the right session. Set on the
   * legacy EventSource path; omitted on the stateless /chat path, where the decision is replayed
   * with the client-carried history instead of a server session.
   */
  sessionId?: string;
  requestId: string;
  toolName: string;
  toolInput: Record<string, any>;
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
  /** A prompt queued to seed the chat input the next time it becomes visible (null when none) */
  pendingPrompt: string | null;
  /** A tool call awaiting the user's Yes/No decision, or null */
  pendingPermission: PermissionRequest | null;
}

export interface AssistantAgentActions {
  /** Open the Assistant panel */
  openPanel: () => void;
  /** Close the Assistant panel */
  closePanel: () => void;
  /** Send a message to Assistant */
  sendMessage: (message: string) => void;
  /** Queue a prompt to seed the chat input the next time it's visible (survives the setup wizard) */
  prefillPrompt: (prompt: string) => void;
  /** Clear any queued prompt */
  clearPendingPrompt: () => void;
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
  /** Answer the pending tool-call permission prompt */
  respondToPermission: (allow: boolean) => void;
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
 * Request body for the stateless POST /chat endpoint (client-carried history).
 * The full conversation history travels with the client and is resent each turn,
 * so the server holds no per-session state.
 */
export interface ChatRequest {
  message: string;
  experiment_id?: string;
  context?: KnownAssistantContext & Record<string, unknown>;
  /** JSON-encoded conversation history carried by the client; omitted on the first turn. */
  conversation_history?: string;
  /**
   * tool_call_id -> decision, sent when resuming a turn paused at a permission prompt. The
   * provider applies it to the matching pending tool_call in the carried history.
   */
  tool_decisions?: Record<string, 'allow' | 'deny'>;
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
  /** Whether this provider carries conversation history client-side and streams statelessly via POST /chat. */
  client_carries_history?: boolean;
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
