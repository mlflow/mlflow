/**
 * Tests for MLflow Claude Agent SDK integration
 */

import * as mlflow from 'mlflow-tracing';
import { tracedClaudeAgent, extractAgentTokenUsage } from '../src';
import { createAuthProvider } from 'mlflow-tracing/src/auth';

const TEST_TRACKING_URI = 'http://localhost:5000';

// Mock Agent class that simulates the Claude Agent SDK behavior
class MockAgent {
  private apiKey: string;
  private tools: any[];

  constructor(config: { apiKey: string; tools?: any[] }) {
    this.apiKey = config.apiKey;
    this.tools = config.tools || [];
  }

  async run(options: { prompt: string; maxTurns?: number }): Promise<any> {
    // Simulate agent execution with Claude API call
    return {
      content: [{ type: 'text', text: `Response to: ${options.prompt}` }],
      usage: {
        input_tokens: 50,
        output_tokens: 100,
      },
      model: 'claude-3-7-sonnet-20250219',
      stop_reason: 'end_turn',
    };
  }

  async execute(options: { message: string }): Promise<any> {
    return {
      response: `Executed: ${options.message}`,
      usage: {
        input_tokens: 30,
        output_tokens: 60,
      },
    };
  }

  async executeTool(toolName: string, input: any): Promise<any> {
    return {
      result: `Tool ${toolName} executed with input: ${JSON.stringify(input)}`,
    };
  }

  async callTool(options: { name: string; arguments: any }): Promise<any> {
    return {
      output: `Called ${options.name}`,
    };
  }

  get subAgent() {
    return {
      async invoke(prompt: string): Promise<any> {
        return { result: `Sub-agent processed: ${prompt}` };
      },
    };
  }
}

describe('tracedClaudeAgent', () => {
  let experimentId: string;
  let client: mlflow.MlflowClient;

  beforeAll(async () => {
    const authProvider = createAuthProvider({ trackingUri: TEST_TRACKING_URI });
    client = new mlflow.MlflowClient({ trackingUri: TEST_TRACKING_URI, authProvider });

    const experimentName = `claude-agent-test-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    experimentId = await client.createExperiment(experimentName);

    mlflow.init({
      trackingUri: TEST_TRACKING_URI,
      experimentId,
    });
  });

  afterAll(async () => {
    await client.deleteExperiment(experimentId);
  });

  const getLastActiveTrace = async (): Promise<mlflow.Trace> => {
    await mlflow.flushTraces();
    const traceId = mlflow.getLastActiveTraceId();
    const trace = await client.getTrace(traceId!);
    return trace;
  };

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Agent.run() tracing', () => {
    it('should trace agent.run() as AGENT span', async () => {
      const agent = new MockAgent({ apiKey: 'test-key' });
      const tracedAgent = tracedClaudeAgent(agent);

      const result = await tracedAgent.run({
        prompt: 'What is the weather in San Francisco?',
        maxTurns: 3,
      });

      expect(result.content[0].text).toContain('What is the weather');

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');

      const span = trace.data.spans[0];
      expect(span.name).toBe('Agent.run');
      expect(span.spanType).toBe(mlflow.SpanType.AGENT);
      expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
      expect(span.inputs).toHaveProperty('prompt', 'What is the weather in San Francisco?');
      expect(span.outputs).toBeDefined();

      // Check token usage
      const tokenUsage = span.attributes[mlflow.SpanAttributeKey.TOKEN_USAGE];
      expect(tokenUsage).toBeDefined();
      expect(tokenUsage[mlflow.TokenUsageKey.INPUT_TOKENS]).toBe(50);
      expect(tokenUsage[mlflow.TokenUsageKey.OUTPUT_TOKENS]).toBe(100);
      expect(tokenUsage[mlflow.TokenUsageKey.TOTAL_TOKENS]).toBe(150);

      // Check message format
      const messageFormat = span.attributes[mlflow.SpanAttributeKey.MESSAGE_FORMAT];
      expect(messageFormat).toBe('anthropic');
    });

    it('should trace agent.execute() as AGENT span', async () => {
      const agent = new MockAgent({ apiKey: 'test-key' });
      const tracedAgent = tracedClaudeAgent(agent);

      const result = await tracedAgent.execute({
        message: 'Hello, Claude!',
      });

      expect(result.response).toContain('Hello, Claude!');

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');

      const span = trace.data.spans[0];
      expect(span.name).toBe('Agent.execute');
      expect(span.spanType).toBe(mlflow.SpanType.AGENT);
    });
  });

  describe('Tool execution tracing', () => {
    it('should trace executeTool() as TOOL span', async () => {
      const agent = new MockAgent({ apiKey: 'test-key' });
      const tracedAgent = tracedClaudeAgent(agent);

      const result = await tracedAgent.executeTool('weather', { city: 'San Francisco' });

      expect(result.result).toContain('weather');

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');

      const span = trace.data.spans[0];
      expect(span.name).toBe('tool_weather');
      expect(span.spanType).toBe(mlflow.SpanType.TOOL);
      expect(span.inputs).toHaveProperty('tool', 'weather');
    });

    it('should trace callTool() as TOOL span', async () => {
      const agent = new MockAgent({ apiKey: 'test-key' });
      const tracedAgent = tracedClaudeAgent(agent);

      const result = await tracedAgent.callTool({
        name: 'calculator',
        arguments: { operation: 'add', a: 1, b: 2 },
      });

      expect(result.output).toContain('calculator');

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');

      const span = trace.data.spans[0];
      expect(span.name).toBe('tool_calculator');
      expect(span.spanType).toBe(mlflow.SpanType.TOOL);
      expect(span.inputs).toHaveProperty('tool', 'calculator');
      expect(span.inputs).toHaveProperty('arguments');
    });
  });

  describe('Nested object tracing', () => {
    it('should trace methods on nested objects (sub-agents)', async () => {
      const agent = new MockAgent({ apiKey: 'test-key' });
      const tracedAgent = tracedClaudeAgent(agent);

      const result = await tracedAgent.subAgent.invoke('Process this data');

      expect(result.result).toContain('Sub-agent processed');

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');

      const span = trace.data.spans[0];
      expect(span.name).toBe('Agent.invoke');
      expect(span.spanType).toBe(mlflow.SpanType.AGENT);
    });
  });

  describe('Error handling', () => {
    it('should handle errors and mark span as ERROR', async () => {
      const agent = new MockAgent({ apiKey: 'test-key' });

      // Override run to throw an error
      (agent as any).run = async () => {
        throw new Error('API rate limit exceeded');
      };

      const tracedAgent = tracedClaudeAgent(agent);

      await expect(tracedAgent.run({ prompt: 'This should fail' })).rejects.toThrow(
        'API rate limit exceeded',
      );

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('ERROR');

      const span = trace.data.spans[0];
      expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.ERROR);
      expect(span.status.message).toContain('API rate limit exceeded');
    });
  });

  describe('Parent span nesting', () => {
    it('should create nested spans when agent is called within a parent span', async () => {
      const agent = new MockAgent({ apiKey: 'test-key' });
      const tracedAgent = tracedClaudeAgent(agent);

      const result = await mlflow.withSpan(
        async (_span) => {
          const response = await tracedAgent.run({
            prompt: 'Nested agent call',
          });
          return response.content[0].text;
        },
        {
          name: 'parent_operation',
          spanType: mlflow.SpanType.CHAIN,
          inputs: { query: 'Nested agent call' },
        },
      );

      expect(result).toContain('Nested agent call');

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');
      expect(trace.data.spans.length).toBe(2);

      const parentSpan = trace.data.spans[0];
      expect(parentSpan.name).toBe('parent_operation');
      expect(parentSpan.spanType).toBe(mlflow.SpanType.CHAIN);

      const childSpan = trace.data.spans[1];
      expect(childSpan.name).toBe('Agent.run');
      expect(childSpan.spanType).toBe(mlflow.SpanType.AGENT);
    });
  });
});

describe('extractAgentTokenUsage', () => {
  it('should extract token usage from direct usage field', () => {
    const response = {
      usage: {
        input_tokens: 100,
        output_tokens: 200,
      },
    };

    const usage = extractAgentTokenUsage(response);

    expect(usage).toEqual({
      input_tokens: 100,
      output_tokens: 200,
      total_tokens: 300,
    });
  });

  it('should extract token usage from nested response.usage', () => {
    const response = {
      response: {
        usage: {
          input_tokens: 50,
          output_tokens: 75,
        },
      },
    };

    const usage = extractAgentTokenUsage(response);

    expect(usage).toEqual({
      input_tokens: 50,
      output_tokens: 75,
      total_tokens: 125,
    });
  });

  it('should extract token usage from message.usage', () => {
    const response = {
      message: {
        usage: {
          input_tokens: 25,
          output_tokens: 50,
        },
      },
    };

    const usage = extractAgentTokenUsage(response);

    expect(usage).toEqual({
      input_tokens: 25,
      output_tokens: 50,
      total_tokens: 75,
    });
  });

  it('should return undefined for missing usage', () => {
    const response = { content: 'Hello' };

    const usage = extractAgentTokenUsage(response);

    expect(usage).toBeUndefined();
  });

  it('should return undefined for null response', () => {
    const usage = extractAgentTokenUsage(null);

    expect(usage).toBeUndefined();
  });
});
