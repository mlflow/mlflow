/**
 * Type declarations for the OpenClaw plugin SDK.
 *
 * These types represent the OpenClaw plugin-sdk interface that this integration
 * targets. Once OpenClaw publishes official TypeScript types, this file can
 * be removed in favor of the `openclaw` package's own type definitions.
 */

declare module 'openclaw/plugin-sdk' {
  export type OpenClawConfig = Record<string, unknown>;

  export type DiagnosticEventPayload = {
    type: string;
    sessionKey?: string;
    costUsd?: number;
    context?: {
      limit?: number;
      used?: number;
    };
    model?: string;
    provider?: string;
    durationMs?: number;
    usage?: {
      input?: number;
      output?: number;
      cacheRead?: number;
      cacheWrite?: number;
      total?: number;
    };
  };

  export type OpenClawPluginService = {
    id: string;
    start: (ctx: {
      config: unknown;
      logger: {
        info: (message: string) => void;
        warn: (message: string) => void;
      };
    }) => void | Promise<void>;
    stop?: (ctx?: unknown) => void | Promise<void>;
  };

  export type OpenClawPluginApi = {
    pluginConfig?: unknown;
    registerService: (service: OpenClawPluginService) => void;
    registerCli: (
      register: (params: { program: unknown }) => void,
      options?: { commands?: string[] },
    ) => void;
    runtime: {
      config: {
        loadConfig: () => OpenClawConfig;
        writeConfigFile: (cfg: OpenClawConfig) => Promise<void>;
      };
    };
    on: (event: string, handler: (event: unknown, ctx: unknown) => void) => void;
  };

  export function onDiagnosticEvent(handler: (event: DiagnosticEventPayload) => void): () => void;

  export function emptyPluginConfigSchema(): unknown;
}
