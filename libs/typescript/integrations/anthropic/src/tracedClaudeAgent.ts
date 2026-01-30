import {
  startSpan,
  LiveSpan,
  SpanAttributeKey,
  SpanType,
  SpanStatusCode,
  TokenUsage,
} from 'mlflow-tracing';

interface HookInput {
  tool_name: string;
  tool_input: unknown;
  tool_use_id: string;
  tool_response?: unknown;
  error?: string;
  session_id?: string;
  model?: string;
  subtype?: string;
}

type HookCallback = (
  input: HookInput,
  toolUseID: string | undefined,
  options: { signal: AbortSignal },
) => Promise<{ continue: boolean }>;

interface HookCallbackMatcher {
  hooks: HookCallback[];
}

interface SDKMessage {
  type: string;
  subtype?: string;
  session_id?: string;
  model?: string;
  message?: { content?: ContentBlock[] };
  result?: string;
  errors?: string[];
  total_cost_usd?: number;
  num_turns?: number;
  duration_ms?: number;
  usage?: { input_tokens?: number; output_tokens?: number };
  modelUsage?: Record<string, { inputTokens?: number; outputTokens?: number }>;
}

interface ContentBlock {
  type: string;
  text?: string;
}

interface Query extends AsyncGenerator<SDKMessage, void> {
  interrupt(): Promise<void>;
  close(): void;
}

interface QueryParams {
  prompt: string | AsyncIterable<unknown>;
  options?: { hooks?: Record<string, HookCallbackMatcher[]>; [key: string]: unknown };
}

type QueryFunction = (params: QueryParams) => Query;

interface StdoutResult {
  stdout?: string;
  stderr?: string;
}

class TracingContext {
  sessionSpan: LiveSpan | null = null;
  activeToolSpans = new Map<string, LiveSpan>();
  sessionId: string | null = null;
  model: string | null = null;
  messages: Array<{ role: string; content: unknown }> = [];
  lastAssistantContent: ContentBlock[] = [];

  createSessionSpan(prompt: string, options?: Record<string, unknown>) {
    this.messages.push({ role: 'user', content: prompt });
    this.sessionSpan = startSpan({
      name: 'claude_agent_query',
      spanType: SpanType.AGENT,
      inputs: { prompt, options: options ? { ...options, hooks: '[hooks]' } : undefined },
      attributes: { 'claude.sdk': '@anthropic-ai/claude-agent-sdk' },
    });
    return this.sessionSpan;
  }

  setSessionInfo(sessionId: string, model?: string) {
    this.sessionId = sessionId;
    this.model = model ?? null;
    this.sessionSpan?.setAttribute('claude.session_id', sessionId);
    if (model) {
      this.sessionSpan?.setAttribute('claude.model', model);
    }
  }

  captureAssistantMessage(message: SDKMessage) {
    const content = message.message?.content;
    if (!Array.isArray(content)) {
      return;
    }

    const hasType = (t: string) => content.some((b) => b?.type === t);

    if (hasType('thinking') && !hasType('text')) {
      this.lastAssistantContent.push(...content);
    } else {
      this.lastAssistantContent = this.lastAssistantContent.length
        ? [...this.lastAssistantContent, ...content]
        : content;
    }

    this.messages.push({ role: 'assistant', content });
  }

  addToolResult(toolUseId: string, result: unknown) {
    let content: string;
    if (typeof result === 'string') {
      content = result;
    } else if (result && typeof result === 'object' && 'stdout' in result) {
      const r = result as StdoutResult;
      content = String(r.stdout || '') + (r.stderr ? '\n' + r.stderr : '');
    } else {
      content = JSON.stringify(result);
    }
    this.messages.push({
      role: 'user',
      content: [{ type: 'tool_result', tool_use_id: toolUseId, content }],
    });
  }

  createToolSpan(toolName: string, toolInput: unknown, toolUseId: string) {
    const span = startSpan({
      name: `tool_${toolName}`,
      spanType: SpanType.TOOL,
      parent: this.sessionSpan ?? undefined,
      inputs: toolInput as Record<string, unknown>,
      attributes: { 'tool.name': toolName, 'tool.use_id': toolUseId },
    });
    this.activeToolSpans.set(toolUseId, span);
    return span;
  }

  endToolSpan(toolUseId: string, result?: unknown, error?: string) {
    const span = this.activeToolSpans.get(toolUseId);
    if (!span) {
      return;
    }

    if (error) {
      span.setOutputs({ error });
      span.setStatus(SpanStatusCode.ERROR, error);
      this.addToolResult(toolUseId, `Error: ${error}`);
    } else {
      span.setOutputs({ result });
      this.addToolResult(toolUseId, result);
    }
    span.end();
    this.activeToolSpans.delete(toolUseId);
  }

  endSession(result?: SDKMessage | null, error?: string) {
    this.activeToolSpans.forEach((s) => s.end());
    this.activeToolSpans.clear();
    if (!this.sessionSpan) {
      return;
    }

    const usage = extractTokenUsage(result);
    if (usage) {
      this.sessionSpan.setAttribute(SpanAttributeKey.TOKEN_USAGE, usage);
    }

    this.sessionSpan.setAttribute(SpanAttributeKey.MESSAGE_FORMAT, 'anthropic');
    this.sessionSpan.setInputs({ model: this.model, messages: this.messages });

    if (result) {
      if (result.total_cost_usd) {
        this.sessionSpan.setAttribute('claude.total_cost_usd', result.total_cost_usd);
      }
      if (result.num_turns) {
        this.sessionSpan.setAttribute('claude.num_turns', result.num_turns);
      }
      if (result.duration_ms) {
        this.sessionSpan.setAttribute('claude.duration_ms', result.duration_ms);
      }
      if (result.subtype) {
        this.sessionSpan.setAttribute('claude.result_type', result.subtype);
      }
    }

    this.sessionSpan.setOutputs({
      id: this.sessionId,
      type: 'message',
      role: 'assistant',
      model: this.model,
      content: this.lastAssistantContent.length
        ? this.lastAssistantContent
        : [{ type: 'text', text: result?.result || '' }],
      usage,
    });

    if (error) {
      this.sessionSpan.setStatus(SpanStatusCode.ERROR, error);
    }
    this.sessionSpan.end();
    this.sessionSpan = null;
  }
}

export function createTracedQuery(queryFn: QueryFunction): QueryFunction {
  return (params: QueryParams): Query => {
    const ctx = new TracingContext();
    const prompt = typeof params.prompt === 'string' ? params.prompt : '[streaming input]';
    ctx.createSessionSpan(prompt, params.options as Record<string, unknown>);

    const existingHooks = params.options?.hooks || {};
    const tracingHooks = {
      PreToolUse: [
        {
          hooks: [
            (input: HookInput) => {
              ctx.createToolSpan(input.tool_name, input.tool_input, input.tool_use_id);
              return Promise.resolve({ continue: true });
            },
          ],
        },
      ],
      PostToolUse: [
        {
          hooks: [
            (input: HookInput) => {
              ctx.endToolSpan(input.tool_use_id, input.tool_response);
              return Promise.resolve({ continue: true });
            },
          ],
        },
      ],
      PostToolUseFailure: [
        {
          hooks: [
            (input: HookInput) => {
              ctx.endToolSpan(input.tool_use_id, undefined, input.error);
              return Promise.resolve({ continue: true });
            },
          ],
        },
      ],
    };

    const actualQuery = queryFn({
      ...params,
      options: {
        ...params.options,
        hooks: {
          ...existingHooks,
          PreToolUse: [...(existingHooks.PreToolUse || []), ...tracingHooks.PreToolUse],
          PostToolUse: [...(existingHooks.PostToolUse || []), ...tracingHooks.PostToolUse],
          PostToolUseFailure: [
            ...(existingHooks.PostToolUseFailure || []),
            ...tracingHooks.PostToolUseFailure,
          ],
        },
      },
    });

    async function* wrappedGenerator(): AsyncGenerator<SDKMessage, void> {
      let lastResult: SDKMessage | null = null;
      let hasError = false;

      try {
        for await (const message of actualQuery) {
          if (message.type === 'system' && message.subtype === 'init') {
            ctx.setSessionInfo(message.session_id ?? '', message.model);
          }
          if (message.type === 'assistant') {
            ctx.captureAssistantMessage(message);
          }
          if (message.type === 'result') {
            lastResult = message;
            hasError = message.subtype !== 'success';
          }
          yield message;
        }
      } catch (e) {
        ctx.endSession(null, e instanceof Error ? e.message : String(e));
        throw e;
      }

      ctx.endSession(lastResult, hasError ? lastResult?.errors?.join(', ') : undefined);
    }

    const wrapped = wrappedGenerator();

    return new Proxy(wrapped as Query, {
      get(target, prop, receiver) {
        if (prop === Symbol.asyncIterator) {
          return () => wrapped;
        }
        if (prop === 'interrupt' || prop === 'close') {
          return (...args: unknown[]) => {
            ctx.endSession(null, 'interrupted');
            const method = (actualQuery as unknown as Record<string, (...a: unknown[]) => unknown>)[
              prop
            ];
            return method?.(...args);
          };
        }
        if (prop === 'next' || prop === 'return' || prop === 'throw') {
          return Reflect.get(target, prop, receiver);
        }
        if (typeof prop === 'string' && prop in actualQuery) {
          const val = (actualQuery as unknown as Record<string, unknown>)[prop];
          if (typeof val === 'function') {
            return (val as (...args: unknown[]) => unknown).bind(actualQuery) as unknown;
          }
          return val;
        }
        return Reflect.get(target, prop, receiver) as unknown;
      },
    });
  };
}

export function extractTokenUsage(result: unknown): TokenUsage | undefined {
  if (!result || typeof result !== 'object') {
    return undefined;
  }
  const r = result as SDKMessage;

  if (r.usage) {
    return {
      input_tokens: r.usage.input_tokens ?? 0,
      output_tokens: r.usage.output_tokens ?? 0,
      total_tokens: (r.usage.input_tokens ?? 0) + (r.usage.output_tokens ?? 0),
    };
  }

  if (r.modelUsage) {
    let input = 0;
    let output = 0;
    for (const m of Object.values(r.modelUsage)) {
      input += m.inputTokens ?? 0;
      output += m.outputTokens ?? 0;
    }
    if (input || output) {
      return { input_tokens: input, output_tokens: output, total_tokens: input + output };
    }
  }

  return undefined;
}
