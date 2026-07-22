/**
 * Lifecycle of a single tool call: `Running` until the matching tool_result
 * arrives, then `Done`/`Error` depending on whether the tool failed.
 */
export const ToolCallStatus = {
  Running: 'running',
  Done: 'done',
  Error: 'error',
} as const;
export type ToolCallStatus = (typeof ToolCallStatus)[keyof typeof ToolCallStatus];

/**
 * One piece of an assistant turn — a text segment or a tool call — mirroring the
 * "message parts" model chat SDKs use (e.g. the Vercel AI SDK's `UIMessage.parts`;
 * the Anthropic API calls the equivalents "content blocks"). A turn is an ordered
 * list of these, not a single string. They're kept in arrival order so the transcript
 * can show tool calls interleaved with the narration, and so tool results/status
 * (filled in later) render where they happened.
 */
export type AssistantPart =
  | { type: 'text'; text: string }
  | {
      type: 'toolCall';
      toolUseId: string;
      name: string;
      input?: Record<string, any>;
      status?: ToolCallStatus;
      // Normalized tool output (string) once the tool_result arrives.
      result?: string;
    };

/**
 * Result of a tool the assistant called, correlated to its tool call by `toolUseId`.
 */
export interface ToolResultInfo {
  toolUseId: string;
  content: string;
  isError: boolean;
}

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
 * A pending tool-call permission request surfaced to the user for a Yes/No
 * decision when the session is not in full-access mode.
 */
export interface PermissionRequest {
  /** The session that produced this request, so a decision targets the right session */
  sessionId: string;
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

/** One provider as reported by the `/providers` discovery endpoint. */
export interface ProviderInfo {
  name: string;
  display_name: string;
  description: string;
  available: boolean;
  selected: boolean;
  requires_api_key: boolean;
  has_api_key: boolean;
  allows_remote_access: boolean;
  /** Curated model options for simple assistant controls; empty when provider decides. */
  model_options: string[];
}

/** The provider that will serve the next chat, per the `/providers` discovery endpoint. */
export interface ResolvedProviderInfo {
  name: string;
  model: string | null;
  auto_selected: boolean;
  requires_api_key: boolean;
  has_api_key: boolean;
  /** LLM provider behind a gateway endpoint (e.g. 'openai'); null/absent otherwise. */
  model_provider?: string | null;
  /** Curated vendor model choices when resolved to an assistant-managed Gateway endpoint. */
  model_options?: string[];
  /** Concrete vendor model backing an assistant-managed Gateway endpoint. */
  provider_model?: string | null;
}

/** Response of the `/providers` discovery endpoint. */
export interface ProvidersResponse {
  providers: ProviderInfo[];
  resolved: ResolvedProviderInfo | null;
  /** Curated model choices for vendor connections the UI can create through the Gateway. */
  gateway_vendor_options?: Record<string, string[]>;
}

export type AssistantProviderSelection =
  | { kind: 'provider'; name: string; model?: string }
  | {
      kind: 'gateway';
      endpointName: string;
      gatewayVendor?: string;
      providerModel?: string;
      modelOptions?: string[];
      requiresApiKey?: boolean;
      hasApiKey?: boolean;
    };

/**
 * Machine-readable codes carried by stream error events so the UI can map a
 * failure to a recovery action. Mirrors `ErrorCode` in `mlflow/assistant/types.py`.
 */
export const AssistantErrorCode = {
  CliNotInstalled: 'cli_not_installed',
  NotAuthenticated: 'not_authenticated',
  ApiKeyMissing: 'api_key_missing',
  NoProvider: 'no_provider',
} as const;
export type AssistantErrorCode = (typeof AssistantErrorCode)[keyof typeof AssistantErrorCode];

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
  /** Machine-readable code for `error` when the backend classified it (else null) */
  errorCode: string | null;
  /** Current tool usage status (e.g., "Reading file...", "Searching...") */
  currentStatus: string | null;
  /** Active tools being used by the assistant */
  activeTools: ToolUseInfo[];
  /** Whether a provider resolves for this client (explicitly selected or auto-picked default) */
  setupComplete: boolean;
  /** Whether config is being loaded */
  isLoadingConfig: boolean;
  /** Whether the server is running locally (localhost) */
  isLocalServer: boolean;
  /** The provider/model backing the composer selection, or null when nothing resolves */
  activeProvider: ResolvedProviderInfo | null;
  /** All providers this client could use, per discovery (feeds the composer's provider picker) */
  providers: ProviderInfo[];
  /** Curated vendor/model shortcuts that create assistant-managed Gateway LLM Connections. */
  gatewayVendorOptions: Record<string, string[]>;
  /** Whether the resolved provider still needs an API key before the first chat */
  needsApiKey: boolean;
  /** A prompt queued to seed the chat input the next time it becomes visible (null when none) */
  pendingPrompt: string | null;
  /** A tool call awaiting the user's Yes/No decision, or null */
  pendingPermission: PermissionRequest | null;
  /** Whether the Assistant can be used from this client, considering server-side remote-access settings */
  canUseAssistant: boolean;
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
  /** Optimistically switch the active provider (persisted on the next send). */
  selectProvider: (selection: AssistantProviderSelection) => void;
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
  gateway_vendor?: string;
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
  /** Whether the currently selected provider can be used from a non-localhost client */
  remote_access_allowed?: boolean;
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
