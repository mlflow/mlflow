export type OpenClawConfig = Record<string, unknown>;

export type OpenClawPluginConfigSchema = {
  type: string;
  properties?: Record<string, unknown>;
  additionalProperties?: boolean;
};

export type OpenClawPluginService = {
  id: string;
  start: (ctx: {
    config: unknown;
    logger: { info: (message: string) => void; warn: (message: string) => void };
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

export function definePluginEntry(options: {
  id: string;
  name: string;
  description: string;
  register: (api: OpenClawPluginApi) => void;
}): unknown {
  return options;
}

export function emptyPluginConfigSchema(): OpenClawPluginConfigSchema {
  return { type: 'object', additionalProperties: false };
}
