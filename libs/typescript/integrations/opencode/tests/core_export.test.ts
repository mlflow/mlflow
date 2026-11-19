import { createHarness, idle, user, assistant } from './support';
import type { PluginDefinition } from '../src/opencode';

describe('MLflow OpenCode V2 plugin with the MLflow exporter', () => {
  const originalEnv = process.env;
  const originalFetch = global.fetch;

  afterEach(() => {
    process.env = originalEnv;
    global.fetch = originalFetch;
    jest.restoreAllMocks();
  });

  const requestPreviews = (fetch: jest.Mock): string[] =>
    fetch.mock.calls
      .filter(([url]) => String(url).endsWith('/api/3.0/mlflow/traces'))
      .map(([, init]) => JSON.parse((init as RequestInit).body as string))
      .map((body) => body.trace.trace_info.request_preview as string);

  it('moves on after a failed upload without exporting the failed execution again', async () => {
    process.env = {
      ...originalEnv,
      MLFLOW_TRACKING_URI: 'http://localhost:5000',
      MLFLOW_EXPERIMENT_ID: '123',
    };
    const fetch = jest
      .fn()
      .mockResolvedValue(new Response('Unavailable', { status: 503, statusText: 'Unavailable' }));
    global.fetch = fetch;
    const log = jest.spyOn(console, 'error').mockImplementation(() => {});
    const plugin: PluginDefinition = (await import('../src')).default;

    const harness = createHarness([user('Hello'), assistant('Hi')]);
    await harness.start(plugin);
    await harness.send(idle());
    await harness.send(idle());
    harness.context.mockResolvedValue([
      user('Hello'),
      assistant('Hi'),
      user('Next', 3000),
      assistant('Next answer', 3100, 4000),
    ]);
    await harness.send(idle());
    await harness.stop();

    expect(requestPreviews(fetch)).toEqual(['Hello', 'Next']);
    expect(log).toHaveBeenCalledWith(
      expect.stringContaining('Failed to export trace'),
      expect.anything(),
    );
  });
});
