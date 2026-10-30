import type {
  AssistantMessage,
  Cleanup,
  ModelInfo,
  PluginContext,
  PluginDefinition,
  PluginEvent,
  SessionMessage,
  ToolPart,
  UserMessage,
} from '../src/opencode';

export type { SessionMessage, ToolPart };

export const idle = (type = 'session.execution.succeeded'): PluginEvent => ({
  type,
  location: { directory: '/project' },
  data: { sessionID: 'ses_test' },
});

export const user = (text: string, created = 1000): UserMessage => ({
  id: `msg_user_${created}`,
  type: 'user',
  text,
  time: { created },
});

export const assistant = (text: string, created = 1100, completed = 1500): AssistantMessage => ({
  id: `msg_assistant_${created}`,
  type: 'assistant',
  agent: 'build',
  time: { created, completed },
  model: { id: 'claude', providerID: 'anthropic' },
  content: [{ type: 'text', text }],
});

export const tool = (): ToolPart => ({
  type: 'tool',
  id: 'call_read',
  name: 'read',
  time: { created: 1150, ran: 1200, completed: 1400 },
  state: {
    status: 'completed',
    input: { path: 'test.ts' },
    metadata: { title: 'Read file' },
    content: [
      { type: 'text', text: 'File contents' },
      { type: 'file', uri: 'file:///test.png', mime: 'image/png' },
    ],
  },
});

export function createHarness(
  messages: SessionMessage[] = [],
  {
    workspaceID,
    options,
    models = [],
  }: { workspaceID?: string; options?: Record<string, unknown>; models?: ModelInfo[] } = {},
) {
  let wake: (() => void) | undefined;
  const events: PluginEvent[] = [];
  let cleanup: Cleanup | void;
  const context = jest.fn().mockResolvedValue(messages);
  const get = jest.fn().mockResolvedValue({ location: { directory: '/project' } });
  const list = jest.fn().mockResolvedValue({ data: models });
  const subscribe = jest.fn(async function* ({ signal }: { signal: AbortSignal }) {
    while (!signal.aborted) {
      const next = events.shift();
      if (next) {
        yield next;
        continue;
      }
      await new Promise<void>((resolve) => {
        const done = () => {
          signal.removeEventListener('abort', done);
          wake = undefined;
          resolve();
        };
        wake = done;
        signal.addEventListener('abort', done, { once: true });
      });
    }
  });
  const settle = () => new Promise<void>((resolve) => setImmediate(resolve));
  return {
    context,
    get,
    list,
    subscribe,
    settle,
    async start(plugin: PluginDefinition) {
      cleanup = await plugin.setup({
        location: { directory: '/project', workspaceID },
        options,
        session: { context, get },
        model: { list },
        event: { subscribe },
      } as unknown as PluginContext);
    },
    async send(event: unknown) {
      events.push(event as PluginEvent);
      wake?.();
      await settle();
    },
    async stop() {
      await cleanup?.();
      cleanup = undefined;
    },
  };
}
