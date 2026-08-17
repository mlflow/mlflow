/**
 * Module-level registry of handlers for CLIENT-executed assistant tool calls
 * (e.g. `render_custom_view`), decoupling the generic Assistant context from
 * feature-specific code (e.g. the Custom View host, which pulls in the
 * ESM-only `@a2ui` renderer and should not be part of the Assistant's
 * static module graph). A feature registers a handler for the tool name(s)
 * it owns; `AssistantContext` looks the handler up by name when a
 * `client_tool_call` event arrives, invokes it, and reports the result back
 * to resume the stream.
 */

export interface ClientToolResult {
  content: string;
  isError?: boolean;
  /** Whether a structured-output provider may automatically retry with this error. */
  retryable?: boolean;
}

export type ClientToolHandler = (toolInput: Record<string, any>) => Promise<ClientToolResult>;

const handlers = new Map<string, ClientToolHandler>();

/** Register a handler for a client tool name. Returns an unregister function. */
export const registerClientToolHandler = (toolName: string, handler: ClientToolHandler): (() => void) => {
  handlers.set(toolName, handler);
  return () => {
    if (handlers.get(toolName) === handler) {
      handlers.delete(toolName);
    }
  };
};

export const getClientToolHandler = (toolName: string): ClientToolHandler | undefined => handlers.get(toolName);
