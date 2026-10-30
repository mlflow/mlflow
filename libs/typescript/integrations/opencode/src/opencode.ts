/**
 * The subset of the OpenCode V2 plugin API that this integration uses.
 *
 * Declared locally instead of depending on @opencode/plugin: the plugin
 * definition is a plain `{ id, setup }` object, and the SDK pulls a large
 * runtime dependency tree pinned to a single OpenCode release.
 */

export interface Location {
  directory: string;
  workspaceID?: string;
}

interface RequestOptions {
  signal?: AbortSignal;
}

export interface TokenUsage {
  input?: number;
  output?: number;
  reasoning?: number;
  cache?: { read?: number; write?: number };
}

export interface TextPart {
  type: 'text' | 'reasoning';
  text: string;
}

export type ToolContent =
  | { type: 'text'; text: string }
  | { type: 'file'; uri: string; mime?: string; name?: string };

export interface ToolState {
  status?: 'streaming' | 'running' | 'completed' | 'error';
  // Streaming tool calls carry the raw, possibly partial JSON arguments.
  input?: string | Record<string, unknown>;
  content?: ToolContent[];
  error?: { type?: string; message?: string };
  metadata?: Record<string, unknown>;
}

export interface ToolPart {
  type: 'tool';
  id: string;
  name: string;
  time?: { created?: number; ran?: number; completed?: number };
  // Tool calls interrupted before their state was persisted have no state.
  state?: ToolState;
}

export interface UserMessage {
  id: string;
  type: 'user';
  text: string;
  time: { created: number };
}

export interface AssistantMessage {
  id: string;
  type: 'assistant';
  agent?: string;
  model: { id: string; providerID: string };
  content: Array<TextPart | ToolPart>;
  tokens?: TokenUsage;
  /** Cost of this model call in USD. */
  cost?: number;
  time: { created: number; completed?: number };
}

/** Session context entries also include control messages (idle, compaction, ...). */
export type SessionMessage = UserMessage | AssistantMessage | { id: string; type: string };

export interface PluginEvent {
  type: string;
  location?: Location;
  data?: { sessionID?: string; error?: { message?: string } };
}

/** Prices in USD per million tokens. Context-size tiers come after the base price. */
export interface ModelCost {
  tier?: { type: string; size: number };
  input: number;
  output: number;
  cache?: { read?: number; write?: number };
}

export interface ModelInfo {
  id: string;
  providerID: string;
  cost?: ModelCost[];
}

export interface PluginContext {
  readonly location: Location;
  readonly options?: Readonly<Record<string, unknown>>;
  readonly model?: {
    list(
      input?: { location?: { directory?: string } },
      options?: RequestOptions,
    ): Promise<{ data: readonly ModelInfo[] }>;
  };
  readonly event: {
    subscribe(options: { signal: AbortSignal; types?: string[] }): AsyncIterable<PluginEvent>;
  };
  readonly session: {
    get(input: { sessionID: string }, options?: RequestOptions): Promise<{ location: Location }>;
    context(input: { sessionID: string }, options?: RequestOptions): Promise<SessionMessage[]>;
  };
}

export type Cleanup = () => Promise<void> | void;

export interface PluginDefinition {
  id: string;
  setup(context: PluginContext): Promise<Cleanup | void> | Cleanup | void;
}

export function isUserMessage(message: SessionMessage): message is UserMessage {
  return message.type === 'user';
}

export function isAssistantMessage(message: SessionMessage): message is AssistantMessage {
  return message.type === 'assistant';
}
