// Mock types for @opencode-ai/plugin
export interface PluginEvent {
  type: string;
  properties?: Record<string, unknown>;
}

export interface EventParams {
  event: PluginEvent;
}

export interface Hooks {
  event?: (params: EventParams) => Promise<void>;
}

export interface PluginClient {
  session: {
    get: (params: { path: { id: string } }) => Promise<{ data?: unknown }>;
    messages: (params: {
      path: { id: string };
      query: { limit: number };
    }) => Promise<{ data?: unknown[] }>;
  };
}

export interface PluginInput {
  client: PluginClient;
  directory: string;
}

export type Plugin = (input: PluginInput) => Promise<Hooks>;
