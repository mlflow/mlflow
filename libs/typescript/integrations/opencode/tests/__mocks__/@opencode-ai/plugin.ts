// Mock types for @opencode-ai/plugin
export type Plugin = (input: any) => Promise<Hooks>;
export type PluginInput = any;
export interface Hooks {
  event?: (params: { event: any }) => Promise<void>;
}
