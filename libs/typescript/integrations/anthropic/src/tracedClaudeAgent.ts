import type {
  Query,
  SDKMessage,
  SDKResultMessage,
  HookCallbackMatcher,
  HookInput,
} from '@anthropic-ai/claude-agent-sdk';

import {
  startSpan,
  LiveSpan,
  SpanAttributeKey,
  SpanType,
  SpanStatusCode,
  TokenUsage,
} from '@mlflow/core';

// Minimal content block type for tracking assistant messages.
// The full BetaContentBlock lives in @anthropic-ai/sdk, so we keep a
// lightweight stand-in to avoid pulling in a second peer dep path.
interface ContentBlock {
  type: string;
  text?: string;
}

interface StdoutResult {
  stdout?: string;
  stderr?: string;
}

type QueryFunction = (params: {
  prompt: string | AsyncIterable<unknown>;
  options?: { hooks?: Partial<Record<string, HookCallbackMatcher[]>>; [key: string]: unknown };
}) => Query;

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
    });
    return this.sessionSpan;
  }

  setSessionInfo(sessionId: string, model?: string) {
    this.sessionId = sessionId;
    this.model = model ?? null;
  }

  captureAssistantMessage(content: ContentBlock[]) {
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

  endSession(result?: SDKResultMessage | null, error?: string) {
    this.activeToolSpans.forEach((s) => s.end());
    this.activeToolSpans.clear();
    if (!this.sessionSpan) {
      return;
    }

    const usage = extractAgentTokenUsage(result);
    if (usage) {
      this.sessionSpan.setAttribute(SpanAttributeKey.TOKEN_USAGE, usage);
    }

    this.sessionSpan.setAttribute(SpanAttributeKey.MESSAGE_FORMAT, 'anthropic');
    this.sessionSpan.setInputs({ model: this.model, messages: this.messages });

    const resultText = result?.subtype === 'success' ? result.result : undefined;

    this.sessionSpan.setOutputs({
      id: this.sessionId,
      type: 'message',
      role: 'assistant',
      model: this.model,
      content: this.lastAssistantContent.length
        ? this.lastAssistantContent
        : [{ type: 'text', text: resultText || '' }],
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
  return (params: Parameters<QueryFunction>[0]): Query => {
    const ctx = new TracingContext();
    const prompt = typeof params.prompt === 'string' ? params.prompt : '[streaming input]';
    ctx.createSessionSpan(prompt, params.options as Record<string, unknown>);

    const existingHooks = params.options?.hooks || {};
    const tracingHooks = {
      PreToolUse: [
        {
          hooks: [
            (input: HookInput) => {
              if (input.hook_event_name !== 'PreToolUse') {
                return Promise.resolve({ continue: true });
              }
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
              if (input.hook_event_name !== 'PostToolUse') {
                return Promise.resolve({ continue: true });
              }
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
              if (input.hook_event_name !== 'PostToolUseFailure') {
                return Promise.resolve({ continue: true });
              }
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
      let lastResult: SDKResultMessage | null = null;
      let hasError = false;

      try {
        for await (const message of actualQuery) {
          if (message.type === 'system' && message.subtype === 'init') {
            ctx.setSessionInfo(message.session_id ?? '', message.model);
          }
          if (message.type === 'assistant') {
            const content = (message as { message?: { content?: ContentBlock[] } }).message?.content;
            if (Array.isArray(content)) {
              ctx.captureAssistantMessage(content);
            }
          }
          if (message.type === 'result') {
            lastResult = message;
            hasError = lastResult.subtype !== 'success';
          }
          yield message;
        }
      } catch (e) {
        ctx.endSession(null, e instanceof Error ? e.message : String(e));
        throw e;
      }

      const errorMsg =
        hasError && lastResult && 'errors' in lastResult
          ? lastResult.errors.join(', ')
          : undefined;
      ctx.endSession(lastResult, errorMsg);
    }

    const wrapped = wrappedGenerator();

    return new Proxy(actualQuery, {
      get(target, prop, receiver) {
        if (prop === Symbol.asyncIterator) {
          return () => wrapped;
        }
        if (prop === 'next' || prop === 'return' || prop === 'throw') {
          return Reflect.get(wrapped, prop, wrapped);
        }
        if (prop === 'interrupt' || prop === 'close') {
          return (...args: unknown[]) => {
            ctx.endSession(null, 'interrupted');
            const method = Reflect.get(target, prop, receiver) as
              | ((...a: unknown[]) => unknown)
              | undefined;
            return typeof method === 'function' ? method.call(target, ...args) : method;
          };
        }
        // eslint-disable-next-line @typescript-eslint/no-unsafe-return
        return Reflect.get(target, prop, receiver);
      },
    });
  };
}

export function extractAgentTokenUsage(result: unknown): TokenUsage | undefined {
  if (!result || typeof result !== 'object') {
    return undefined;
  }
  const r = result as SDKResultMessage;

  if (r.usage) {
    const { input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens } =
      r.usage;
    const totalInput =
      (input_tokens ?? 0) + (cache_read_input_tokens ?? 0) + (cache_creation_input_tokens ?? 0);
    const totalOutput = output_tokens ?? 0;
    return {
      input_tokens: totalInput,
      output_tokens: totalOutput,
      total_tokens: totalInput + totalOutput,
    };
  }

  if (r.modelUsage) {
    let input = 0;
    let output = 0;
    for (const m of Object.values(r.modelUsage)) {
      input +=
        (m.inputTokens ?? 0) + (m.cacheReadInputTokens ?? 0) + (m.cacheCreationInputTokens ?? 0);
      output += m.outputTokens ?? 0;
    }
    if (input || output) {
      return { input_tokens: input, output_tokens: output, total_tokens: input + output };
    }
  }

  return undefined;
}
