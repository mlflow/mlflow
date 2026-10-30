import type { PluginDefinition } from '../src/opencode';
import {
  createHarness,
  idle,
  user,
  assistant,
  tool,
  type SessionMessage,
  type ToolPart,
} from './support';

jest.mock('@mlflow/core', () => {
  const span = {
    traceId: 'trace-id',
    setAttribute: jest.fn(),
    setOutputs: jest.fn(),
    setStatus: jest.fn(),
    end: jest.fn(),
  };
  return {
    init: jest.fn(),
    startSpan: jest.fn(() => span),
    withSpan: jest.fn((callback: (value: typeof span) => unknown) => callback(span)),
    updateCurrentTrace: jest.fn(),
    flushTraces: jest.fn().mockResolvedValue(undefined),
    SpanType: { LLM: 'LLM', TOOL: 'TOOL', AGENT: 'AGENT' },
    SpanAttributeKey: { TOKEN_USAGE: 'token_usage' },
    SpanStatusCode: { OK: 'STATUS_CODE_OK', ERROR: 'STATUS_CODE_ERROR' },
    TokenUsageKey: {
      INPUT_TOKENS: 'input_tokens',
      OUTPUT_TOKENS: 'output_tokens',
      TOTAL_TOKENS: 'total_tokens',
      CACHE_READ_INPUT_TOKENS: 'cache_read_input_tokens',
      CACHE_CREATION_INPUT_TOKENS: 'cache_creation_input_tokens',
    },
    NO_OP_SPAN_TRACE_ID: 'no-op-span-trace-id',
  };
});

const COMPLETION_EVENTS = [
  'session.execution.succeeded',
  'session.execution.failed',
  'session.execution.interrupted',
];

describe('MLflow OpenCode V2 plugin', () => {
  const originalEnv = process.env;
  let plugin: PluginDefinition;
  let mlflow: typeof import('@mlflow/core');
  let harness: ReturnType<typeof createHarness>;

  beforeEach(async () => {
    jest.resetModules();
    process.env = {
      ...originalEnv,
      MLFLOW_TRACKING_URI: 'http://localhost:5000',
      MLFLOW_EXPERIMENT_ID: '123',
      USER: 'test-user',
    };
    delete process.env.MLFLOW_OPENCODE_DEBUG;
    mlflow = await import('@mlflow/core');
    plugin = (await import('../src')).default;
    harness = createHarness([user('Hello'), assistant('Hi')]);
  });

  afterEach(async () => {
    await harness.stop();
    process.env = originalEnv;
    jest.restoreAllMocks();
  });

  interface StubSpanOptions {
    name: string;
    spanType?: string;
    startTimeNs?: number;
    inputs?: { model?: string; messages?: unknown[]; prompt?: string } & Record<string, unknown>;
    attributes?: Record<string, unknown>;
  }
  interface StubSpan {
    traceId: string;
    setAttribute: jest.Mock;
    setOutputs: jest.Mock;
    setStatus: jest.Mock;
    end: jest.Mock;
  }
  const startSpanMock = () => mlflow.startSpan as jest.Mock;
  const spans = (name: string): StubSpanOptions[] =>
    startSpanMock()
      .mock.calls.map((args) => args[0] as StubSpanOptions)
      .filter((options) => options.name === name);
  const span = (): StubSpan => startSpanMock().mock.results[0].value as StubSpan;

  it('default-exports a V2 definition with a stable ID and setup function', async () => {
    expect(plugin.id).toBe('mlflow.tracing');
    expect(typeof plugin.setup).toBe('function');
    expect((await import('../src')).MLflowTracingPlugin).toBe(plugin);
    // Newer OpenCode V1 releases load this entry too and call server().
    expect((plugin as unknown as { server: unknown }).server).toBe(
      (await import('../src/v1')).MLflowTracingPlugin,
    );
    await harness.start(plugin);
    expect(harness.subscribe).toHaveBeenCalledWith({
      signal: expect.any(AbortSignal),
      types: COMPLETION_EVENTS,
    });
    expect(mlflow.init).not.toHaveBeenCalled();
  });

  it('prefers plugin options over environment variables', async () => {
    harness = createHarness([user('Hello'), assistant('Hi')], {
      options: { trackingUri: 'https://mlflow.example.com', experimentId: 7 },
    });
    await harness.start(plugin);
    await harness.send(idle());
    expect(mlflow.init).toHaveBeenCalledWith({
      trackingUri: 'https://mlflow.example.com',
      experimentId: '7',
    });
  });

  it('traces with plugin options when the environment variables are not set', async () => {
    delete process.env.MLFLOW_TRACKING_URI;
    delete process.env.MLFLOW_EXPERIMENT_ID;
    harness = createHarness([user('Hello'), assistant('Hi')], {
      options: { trackingUri: 'http://localhost:5000', experimentId: '123' },
    });
    await harness.start(plugin);
    await harness.send(idle());
    expect(mlflow.withSpan).toHaveBeenCalledTimes(1);
  });

  it('enables debug logging through the debug option', async () => {
    const log = jest.spyOn(console, 'error').mockImplementation(() => {});
    harness = createHarness([user('Hello'), assistant('Hi')], { options: { debug: true } });
    await harness.start(plugin);
    await harness.send(idle());
    expect(log).toHaveBeenCalledWith('[mlflow] Created trace: trace-id');
  });

  it.each(['MLFLOW_TRACKING_URI', 'MLFLOW_EXPERIMENT_ID'])(
    'does not trace without %s',
    async (key) => {
      delete process.env[key];
      await harness.start(plugin);
      await harness.send(idle());
      expect(mlflow.init).not.toHaveBeenCalled();
      expect(harness.context).not.toHaveBeenCalled();
    },
  );

  it('initializes lazily and reads the V2 session context without a V1 response envelope', async () => {
    await harness.start(plugin);
    await harness.send(idle());
    expect(mlflow.init).toHaveBeenCalledWith({
      trackingUri: 'http://localhost:5000',
      experimentId: '123',
    });
    expect(harness.context).toHaveBeenCalledWith(
      { sessionID: 'ses_test' },
      { signal: expect.any(AbortSignal) },
    );
    expect(mlflow.withSpan).toHaveBeenCalledWith(
      expect.any(Function),
      expect.objectContaining({
        name: 'opencode_conversation',
        spanType: 'AGENT',
        inputs: { prompt: 'Hello' },
      }),
    );
    expect(spans('llm_call')[0]?.inputs?.model).toBe('anthropic/claude');
    expect(mlflow.updateCurrentTrace).toHaveBeenCalledWith({
      metadata: { 'mlflow.trace.session': 'ses_test', 'mlflow.trace.user': 'test-user' },
      requestPreview: 'Hello',
      responsePreview: 'Hi',
    });
    expect(mlflow.flushTraces).toHaveBeenCalledTimes(1);
  });

  it('ignores non-completion events', async () => {
    await harness.start(plugin);
    await harness.send({ ...idle(), type: 'server.connected', data: {} });
    expect(harness.context).not.toHaveBeenCalled();
  });

  it.each([{ directory: '/other-project' }, { directory: '/project', workspaceID: 'wrk_other' }])(
    'does not export events from another location: %j',
    async (location) => {
      await harness.start(plugin);
      await harness.send({ ...idle(), location });
      expect(harness.context).not.toHaveBeenCalled();
    },
  );

  it.each([
    'session.execution.succeeded',
    'session.execution.failed',
    'session.execution.interrupted',
  ])('traces native %s events without location', async (type) => {
    await harness.start(plugin);
    await harness.send({ ...idle(), type, location: undefined });
    expect(harness.get).toHaveBeenCalledWith(
      { sessionID: 'ses_test' },
      { signal: expect.any(AbortSignal) },
    );
    expect(mlflow.withSpan).toHaveBeenCalledTimes(1);
  });

  it('does not export a locationless completion for a session in another project', async () => {
    harness.get.mockResolvedValue({ location: { directory: '/other-project' } });
    await harness.start(plugin);
    await harness.send({ ...idle(), location: undefined });
    expect(harness.context).not.toHaveBeenCalled();
    expect(mlflow.withSpan).not.toHaveBeenCalled();
  });

  it('ignores the legacy session.idle event rather than depending on it', async () => {
    await harness.start(plugin);
    await harness.send({ ...idle(), type: 'session.idle' });
    expect(harness.context).not.toHaveBeenCalled();
  });

  it('matches workspace identity as well as directory', async () => {
    harness = createHarness([user('Workspace prompt'), assistant('Workspace response')], {
      workspaceID: 'wrk_test',
    });
    await harness.start(plugin);
    await harness.send(idle());
    expect(harness.context).not.toHaveBeenCalled();
    await harness.send({ ...idle(), location: { directory: '/project', workspaceID: 'wrk_test' } });
    expect(mlflow.withSpan).toHaveBeenCalledTimes(1);
  });

  it('captures reasoning, model identity, token usage, and timestamps', async () => {
    harness.context.mockResolvedValue([
      user('Think', 1000),
      {
        ...assistant('Answer', 1100, 1800),
        model: { id: 'reasoner', providerID: 'openai' },
        tokens: { input: 100, output: 20, reasoning: 10, cache: { read: 5, write: 2 } },
        content: [
          { type: 'reasoning', text: 'Reasoning' },
          { type: 'text', text: 'Answer' },
        ],
      },
    ]);
    await harness.start(plugin);
    await harness.send(idle());
    expect(spans('llm_call')[0]).toMatchObject({
      startTimeNs: 1100e6,
      inputs: { model: 'openai/reasoner' },
    });
    // OpenCode excludes cache tokens from input and reasoning tokens from output.
    expect(span().setAttribute).toHaveBeenCalledWith('token_usage', {
      input_tokens: 107,
      output_tokens: 30,
      total_tokens: 137,
      cache_read_input_tokens: 5,
      cache_creation_input_tokens: 2,
    });
    expect(span().setOutputs).toHaveBeenCalledWith({
      choices: [
        {
          message: {
            role: 'assistant',
            content: 'Answer',
            reasoning: 'Reasoning',
          },
        },
      ],
    });
    expect(span().end).toHaveBeenCalledWith({ endTimeNs: 1800e6 });
    expect(mlflow.withSpan).toHaveBeenCalledWith(
      expect.any(Function),
      expect.objectContaining({ startTimeNs: 1000e6 }),
    );
  });

  it('traces the latest turn and its history on the first export', async () => {
    harness.context.mockResolvedValue([
      user('Turn one', 1000),
      assistant('One', 1100, 1500),
      user('Turn two', 2000),
      assistant('Two', 2100, 2500),
    ]);
    await harness.start(plugin);
    await harness.send(idle());
    expect(spans('llm_call')).toHaveLength(1);
    expect(spans('llm_call')[0]?.startTimeNs).toBe(2100e6);
    expect(spans('llm_call')[0]?.inputs?.messages).toContainEqual({
      role: 'assistant',
      content: 'One',
    });
    expect(mlflow.updateCurrentTrace).toHaveBeenCalledWith(
      expect.objectContaining({ requestPreview: 'Turn two' }),
    );
  });

  it('records provider and model attributes per assistant message', async () => {
    harness.context.mockResolvedValue([
      user('Switch', 1000),
      { ...assistant('From openai', 1100, 1500), model: { id: 'gpt', providerID: 'openai' } },
      { ...assistant('From gemini', 2100, 2500), model: { id: 'flash', providerID: 'google' } },
    ]);
    await harness.start(plugin);
    await harness.send(idle());
    expect(spans('llm_call')[0]?.attributes).toMatchObject({ model: 'gpt', provider: 'openai' });
    expect(spans('llm_call')[1]?.attributes).toMatchObject({ model: 'flash', provider: 'google' });
  });

  it('omits the reasoning field when no reasoning parts exist', async () => {
    await harness.start(plugin);
    await harness.send(idle());
    expect(span().setOutputs).toHaveBeenCalledWith({
      choices: [{ message: { role: 'assistant', content: 'Hi' } }],
    });
  });

  it('traces reasoning-only assistant messages without a response preview', async () => {
    harness.context.mockResolvedValue([
      user('Think'),
      { ...assistant(''), content: [{ type: 'reasoning', text: 'Reasoning only' }] },
    ]);
    await harness.start(plugin);
    await harness.send(idle());
    expect(spans('llm_call')).toHaveLength(1);
    expect(span().setOutputs).toHaveBeenCalledWith({
      choices: [
        {
          message: {
            role: 'assistant',
            content: null,
            reasoning: 'Reasoning only',
          },
        },
      ],
    });
    expect((mlflow.updateCurrentTrace as jest.Mock).mock.calls[0][0]).not.toHaveProperty(
      'responsePreview',
    );
  });

  it('traces tool-only responses, V2 structured results, and tool timing', async () => {
    harness.context.mockResolvedValue([
      user('Read'),
      { ...assistant(''), content: [tool()] },
      assistant('Done'),
    ]);
    await harness.start(plugin);
    await harness.send(idle());
    expect(spans('llm_call')).toHaveLength(2);
    expect(spans('tool_read')[0]).toMatchObject({
      spanType: 'TOOL',
      startTimeNs: 1200e6,
      inputs: { path: 'test.ts' },
      attributes: { tool_name: 'read', tool_id: 'call_read', status: 'completed' },
    });
    expect(span().setOutputs).toHaveBeenCalledWith({
      result: 'File contents\n[file: file:///test.png]',
      title: 'Read file',
    });
    expect(span().setOutputs).toHaveBeenCalledWith({
      choices: [
        {
          message: {
            role: 'assistant',
            content: null,
            tool_calls: [
              {
                id: 'call_read',
                type: 'function',
                function: { name: 'read', arguments: '{"path":"test.ts"}' },
              },
            ],
          },
        },
      ],
    });
    expect(span().end).toHaveBeenCalledWith({ endTimeNs: 1400e6 });
    expect(spans('llm_call')[1]?.inputs?.messages).toContainEqual({
      role: 'tool',
      tool_call_id: 'call_read',
      content: 'File contents\n[file: file:///test.png]',
    });
  });

  it('omits the tool title when OpenCode does not set one', async () => {
    const untitled = { ...tool(), state: { ...tool().state, metadata: { truncated: false } } };
    harness.context.mockResolvedValue([user('Read'), { ...assistant(''), content: [untitled] }]);
    await harness.start(plugin);
    await harness.send(idle());
    expect(span().setOutputs).toHaveBeenCalledWith({
      result: 'File contents\n[file: file:///test.png]',
    });
  });

  it('creates tool spans for edit, write, and bash calls', async () => {
    const renamed = (id: string, name: string): ToolPart => ({ ...tool(), id, name });
    harness.context.mockResolvedValue([
      user('Apply changes'),
      {
        ...assistant(''),
        content: [
          renamed('call_edit', 'edit'),
          renamed('call_write', 'write'),
          renamed('call_bash', 'bash'),
        ],
      },
    ]);
    await harness.start(plugin);
    await harness.send(idle());
    for (const name of ['edit', 'write', 'bash']) {
      expect(spans(`tool_${name}`)).toHaveLength(1);
    }
  });

  it('skips assistant messages without content parts', async () => {
    harness.context.mockResolvedValue([
      user('Hello'),
      { ...assistant(''), content: [] },
      assistant('Answer', 1200, 1600),
    ]);
    await harness.start(plugin);
    await harness.send(idle());
    expect(spans('llm_call')).toHaveLength(1);
    expect(spans('llm_call')[0]?.startTimeNs).toBe(1200e6);
  });

  it('handles a tool call without state', async () => {
    const bare = {
      type: 'tool',
      id: 'call_bare',
      name: 'bash',
      time: { created: 1150, ran: 1200, completed: 1300 },
    } as unknown as ToolPart;
    harness.context.mockResolvedValue([user('Run'), { ...assistant(''), content: [bare] }]);
    await harness.start(plugin);
    await harness.send(idle());
    expect(spans('tool_bash')[0]).toMatchObject({
      inputs: {},
      attributes: expect.objectContaining({ tool_name: 'bash', status: 'unknown' }),
    });
  });

  it('records structured tool errors', async () => {
    const failed = {
      ...tool(),
      state: { status: 'error', input: {}, error: { type: 'test', message: 'Command failed' } },
    };
    harness.context.mockResolvedValue([user('Run'), { ...assistant(''), content: [failed] }]);
    await harness.start(plugin);
    await harness.send(idle());
    expect(spans('tool_read')[0]?.attributes?.status).toBe('error');
    expect(span().setOutputs).toHaveBeenCalledWith({ error: 'Command failed' });
    expect(span().end).toHaveBeenCalledWith({ endTimeNs: 1400e6, status: 'STATUS_CODE_ERROR' });
  });

  const pricedMessages = (): SessionMessage[] => [
    user('Run'),
    {
      ...assistant('', 1100, 1500),
      content: [tool()],
      // Base prices: 600 * 1 + 200 * 0.5 + 100 * 2 = 900 input, 100 * 3 = 300 output.
      tokens: { input: 600, output: 60, reasoning: 40, cache: { read: 200, write: 100 } },
      cost: 0.0012,
    },
    { ...assistant('Done', 1600, 1900), tokens: { input: 100, output: 100 }, cost: 0.0004 },
  ];
  const haikuPrices = [
    {
      id: 'claude',
      providerID: 'anthropic',
      cost: [
        { tier: { type: 'context', size: 200000 }, input: 10, output: 10 },
        { input: 1, output: 3, cache: { read: 0.5, write: 2 } },
      ],
    },
  ];
  const costCalls = () =>
    span()
      .setAttribute.mock.calls.filter(([key]) => key === 'mlflow.llm.cost')
      .map(([, value]) => value as Record<string, number>);
  const traceCost = () =>
    JSON.parse(
      (mlflow.updateCurrentTrace as jest.Mock).mock.calls[0][0].metadata['mlflow.trace.cost'],
    ) as Record<string, number>;

  it('splits the cost of each model call by the model base prices', async () => {
    harness = createHarness(pricedMessages(), { models: haikuPrices });
    await harness.start(plugin);
    await harness.send(idle());
    expect(harness.list).toHaveBeenCalledWith(
      { location: { directory: '/project' } },
      { signal: expect.any(AbortSignal) },
    );
    const [first, second] = costCalls();
    expect(first.input_cost).toBeCloseTo(0.0009);
    expect(first.output_cost).toBeCloseTo(0.0003);
    expect(first.total_cost).toBe(0.0012);
    expect(second.input_cost).toBeCloseTo(0.0001);
    expect(second.output_cost).toBeCloseTo(0.0003);
    expect(traceCost().input_cost).toBeCloseTo(0.001);
    expect(traceCost().output_cost).toBeCloseTo(0.0006);
    expect(traceCost().total_cost).toBeCloseTo(0.0016);
  });

  it('records only the total cost when model prices are unknown', async () => {
    harness = createHarness(pricedMessages());
    await harness.start(plugin);
    await harness.send(idle());
    expect(costCalls()).toEqual([{ total_cost: 0.0012 }, { total_cost: 0.0004 }]);
    expect(traceCost()).toEqual({ total_cost: expect.closeTo(0.0016) });
  });

  it('records zero cost for free models without looking up prices', async () => {
    harness.context.mockResolvedValue([user('Hello'), { ...assistant('Hi'), cost: 0 }]);
    await harness.start(plugin);
    await harness.send(idle());
    expect(harness.list).not.toHaveBeenCalled();
    expect(costCalls()).toEqual([{ input_cost: 0, output_cost: 0, total_cost: 0 }]);
  });

  it('looks up model prices once per model', async () => {
    harness = createHarness(pricedMessages(), { models: haikuPrices });
    await harness.start(plugin);
    await harness.send(idle());
    harness.context.mockResolvedValue([
      ...pricedMessages(),
      user('Again', 3000),
      { ...assistant('Again', 3100, 3500), cost: 0.1 },
      { ...assistant('Other', 3600, 3900), model: { id: 'gpt', providerID: 'openai' }, cost: 0.1 },
    ]);
    await harness.send(idle());
    await harness.send({ ...idle(), data: { sessionID: 'ses_other' } });
    expect(harness.list).toHaveBeenCalledTimes(2);
  });

  it('still traces when the model price lookup fails', async () => {
    harness = createHarness(pricedMessages(), { models: haikuPrices });
    harness.list.mockRejectedValue(new Error('Unavailable'));
    await harness.start(plugin);
    await harness.send(idle());
    expect(mlflow.withSpan).toHaveBeenCalledTimes(1);
    expect(costCalls()[0]).toEqual({ total_cost: 0.0012 });
  });

  it('omits cost when OpenCode does not report it', async () => {
    await harness.start(plugin);
    await harness.send(idle());
    expect(span().setAttribute).not.toHaveBeenCalledWith('mlflow.llm.cost', expect.anything());
    expect((mlflow.updateCurrentTrace as jest.Mock).mock.calls[0][0].metadata).not.toHaveProperty(
      'mlflow.trace.cost',
    );
  });

  it('marks failed executions with an error status', async () => {
    await harness.start(plugin);
    await harness.send({
      ...idle('session.execution.failed'),
      data: { sessionID: 'ses_test', error: { type: 'provider', message: 'Rate limited' } },
    });
    expect(span().setStatus).toHaveBeenCalledWith('STATUS_CODE_ERROR', 'Rate limited');
    expect(span().end).toHaveBeenCalledWith({
      outputs: { response: 'Hi', status: 'failed', error: 'Rate limited' },
      endTimeNs: 1500e6,
    });
  });

  it('records interrupted executions without an error status', async () => {
    await harness.start(plugin);
    await harness.send(idle('session.execution.interrupted'));
    expect(span().setStatus).not.toHaveBeenCalled();
    expect(span().end).toHaveBeenCalledWith({
      outputs: { response: 'Hi', status: 'interrupted' },
      endTimeNs: 1500e6,
    });
  });

  it('warns once when the MLflow tracer is inactive after a plugin reload', async () => {
    const warning = jest.spyOn(console, 'error').mockImplementation(() => {});
    (mlflow.withSpan as jest.Mock).mockImplementation(() => 'no-op-span-trace-id');
    await harness.start(plugin);
    await harness.send(idle());
    harness.context.mockResolvedValue([user('Hello'), assistant('Hi'), user('Next', 3000)]);
    await harness.send(idle());
    expect(warning).toHaveBeenCalledTimes(1);
    expect(warning).toHaveBeenCalledWith(expect.stringContaining('opencode service restart'));
  });

  it('handles a streaming tool with partial JSON input', async () => {
    harness.context.mockResolvedValue([
      user('Run'),
      {
        ...assistant(''),
        content: [{ ...tool(), state: { status: 'streaming', input: '{"path":' } }],
      },
    ]);
    await harness.start(plugin);
    await harness.send(idle());
    expect(spans('tool_read')[0].inputs).toEqual({ arguments: '{"path":' });
  });

  it('preserves previous text, reasoning, and tool history in the current turn', async () => {
    harness.context.mockResolvedValue([
      user('First'),
      {
        ...assistant('First answer'),
        content: [
          { type: 'text', text: 'First answer' },
          { type: 'reasoning', text: 'Thoughts' },
          tool(),
        ],
      },
      user('Second'),
      assistant('Second answer'),
    ]);
    await harness.start(plugin);
    await harness.send(idle());
    const history = spans('llm_call')[0]?.inputs?.messages;
    expect(history).toContainEqual({
      role: 'assistant',
      content: 'First answer',
      reasoning: 'Thoughts',
    });
    expect(history).toContainEqual({ role: 'user', content: 'Second' });
    expect(history).toContainEqual({
      role: 'tool',
      tool_call_id: 'call_read',
      content: 'File contents\n[file: file:///test.png]',
    });
  });

  it.each([
    { messages: [] },
    { messages: [assistant('No user')] },
    { messages: [user(''), assistant('Empty prompt')] },
    { messages: [{ id: 'msg_idle', type: 'idle', time: { created: 2000 }, outcome: 'succeeded' }] },
  ])('does not create a trace without a user prompt: %j', async ({ messages }) => {
    harness.context.mockResolvedValue(messages);
    await harness.start(plugin);
    await harness.send(idle());
    expect(mlflow.withSpan).not.toHaveBeenCalled();
  });

  it('ignores non-conversation control messages', async () => {
    harness.context.mockResolvedValue([
      {
        id: 'msg_model',
        type: 'model-switched',
        model: { id: 'claude', providerID: 'anthropic' },
        time: { created: 500 },
      },
      user('Hello'),
      assistant('Hi'),
      { id: 'msg_idle', type: 'idle', time: { created: 2000 }, outcome: 'succeeded' },
    ]);
    await harness.start(plugin);
    await harness.send(idle());
    expect(spans('llm_call')).toHaveLength(1);
  });

  it('truncates request and response previews at 1000 characters', async () => {
    harness.context.mockResolvedValue([user('x'.repeat(1500)), assistant('y'.repeat(1500))]);
    await harness.start(plugin);
    await harness.send(idle());
    expect(mlflow.updateCurrentTrace).toHaveBeenCalledWith(
      expect.objectContaining({
        requestPreview: 'x'.repeat(1000),
        responsePreview: 'y'.repeat(1000),
      }),
    );
  });

  it('does not export a turn twice', async () => {
    await harness.start(plugin);
    await harness.send(idle());
    await harness.send(idle());
    expect(mlflow.withSpan).toHaveBeenCalledTimes(1);
  });

  it('traces new messages after an earlier idle event', async () => {
    await harness.start(plugin);
    await harness.send(idle());
    harness.context.mockResolvedValue([
      user('Hello'),
      assistant('Hi'),
      user('Next', 3000),
      assistant('Next answer', 3100, 4000),
    ]);
    await harness.send(idle());
    expect(mlflow.withSpan).toHaveBeenCalledTimes(2);
    expect(mlflow.withSpan).toHaveBeenLastCalledWith(
      expect.any(Function),
      expect.objectContaining({ inputs: { prompt: 'Next' } }),
    );
  });

  it('traces a shorter context after compaction, including equal-length replacements', async () => {
    harness.context.mockResolvedValue([
      user('Old'),
      assistant('Old answer'),
      user('Latest', 3000),
      assistant('Latest answer', 3100, 4000),
    ]);
    await harness.start(plugin);
    await harness.send(idle());
    harness.context.mockResolvedValue([
      user('After compaction', 5000),
      assistant('New answer', 5100, 6000),
    ]);
    await harness.send(idle());
    harness.context.mockResolvedValue([
      user('Same length', 7000),
      assistant('Another answer', 7100, 8000),
    ]);
    await harness.send(idle());
    expect(mlflow.withSpan).toHaveBeenCalledTimes(3);
  });

  it('retries after a context fetch failure', async () => {
    harness.context.mockRejectedValueOnce(new Error('Unavailable'));
    await harness.start(plugin);
    await harness.send(idle());
    expect(mlflow.withSpan).not.toHaveBeenCalled();
    await harness.send(idle());
    expect(mlflow.withSpan).toHaveBeenCalledTimes(1);
  });

  it('does not advance the checkpoint after an export failure', async () => {
    (mlflow.flushTraces as jest.Mock).mockRejectedValueOnce(new Error('Unavailable'));
    await harness.start(plugin);
    await harness.send(idle());
    await harness.send(idle());
    expect(mlflow.flushTraces).toHaveBeenCalledTimes(2);
    await harness.send(idle());
    expect(mlflow.flushTraces).toHaveBeenCalledTimes(2);
  });

  it('serializes duplicate idle work without blocking event consumption', async () => {
    let resolve!: (messages: SessionMessage[]) => void;
    harness.context.mockImplementationOnce(
      () =>
        new Promise<SessionMessage[]>((done) => {
          resolve = done;
        }),
    );
    await harness.start(plugin);
    await harness.send(idle());
    await harness.send(idle());
    expect(harness.context).toHaveBeenCalledTimes(1);
    resolve([user('Hello'), assistant('Hi')]);
    await harness.settle();
    expect(harness.context).toHaveBeenCalledTimes(2);
    expect(mlflow.withSpan).toHaveBeenCalledTimes(1);
  });

  it('uses separate checkpoints for each plugin instance', async () => {
    await harness.start(plugin);
    await harness.send(idle());
    await harness.stop();
    harness = createHarness([user('Hello'), assistant('Hi')]);
    await harness.start(plugin);
    await harness.send(idle());
    expect(mlflow.withSpan).toHaveBeenCalledTimes(2);
  });

  it('resubscribes after a stream failure', async () => {
    harness.subscribe.mockImplementationOnce(async function* () {
      yield* [];
      await Promise.resolve();
      throw new Error('Stream disconnected');
    });
    await harness.start(plugin);
    await new Promise((resolve) => setTimeout(resolve, 1100));
    expect(harness.subscribe).toHaveBeenCalledTimes(2);
    await harness.send(idle());
    expect(mlflow.withSpan).toHaveBeenCalledTimes(1);
  });

  it('cancels an in-flight context request during unload', async () => {
    harness.context.mockImplementationOnce(
      (_input: unknown, { signal }: { signal: AbortSignal }) =>
        new Promise((_resolve, reject) => {
          signal.addEventListener('abort', () => reject(new Error('Aborted')), { once: true });
        }),
    );
    await harness.start(plugin);
    await harness.send(idle());
    expect(harness.context).toHaveBeenCalledTimes(1);
    await harness.stop();
    expect(mlflow.withSpan).not.toHaveBeenCalled();
  });

  it('aborts the event subscription on unload and ignores later events', async () => {
    await harness.start(plugin);
    const signal = harness.subscribe.mock.calls[0][0].signal;
    await harness.stop();
    expect(signal.aborted).toBe(true);
    await harness.send(idle());
    expect(harness.context).not.toHaveBeenCalled();
  });
});
