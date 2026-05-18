/**
 * Wraps the Claude Agent SDK `query()` function so each call produces an
 * MLflow trace. The wrapper sets `forwardSubagentText: true` on options by
 * default so sub-agent activity flows through the parent message stream and
 * can be attributed via `parent_tool_use_id`.
 *
 * The actual span construction lives in LiveTracingContext, which mirrors the
 * span tree produced by processTranscript (CLI Stop-hook path).
 */

import type { Query, SDKMessage } from '@anthropic-ai/claude-agent-sdk';

import {
  LiveTracingContext,
  type SDKAssistantLike,
  type SDKResultLike,
  type SDKSystemInit,
  type SDKUserLike,
} from './liveTracing.js';

type QueryOptions = {
  forwardSubagentText?: boolean;
  [key: string]: unknown;
};

type QueryParams = {
  prompt: string | AsyncIterable<unknown>;
  options?: QueryOptions;
};

type QueryFunction = (params: QueryParams) => Query;

/**
 * Wrap a Claude Agent SDK `query` function so each invocation produces an
 * MLflow trace. The wrapped query has the same call signature and return
 * type as the original; tracing is fully transparent to the caller.
 */
export function createTracedQuery(queryFn: QueryFunction): QueryFunction {
  return (params: QueryParams): Query => {
    const ctx = new LiveTracingContext(initialPromptFor(params.prompt), params.options);

    // For AsyncIterable prompts, wrap so user messages are observed for prompt
    // capture without consuming them — the SDK still receives the original
    // sequence. The wrapper records text content on the root span's inputs.
    const prompt =
      typeof params.prompt === 'string'
        ? params.prompt
        : capturingPromptIterable(params.prompt, (text) => ctx.appendPromptText(text));

    // Force forwardSubagentText: true so sub-agent inner messages flow
    // through the parent stream. Without it, the SDK only emits tool_use and
    // tool_result blocks from sub-agents (heartbeat), which strips LLM and
    // nested tool spans from the resulting trace. Forwarding is local-only
    // (no network cost), so we always enable it when the wrapper is in use.
    const options: QueryOptions = {
      ...params.options,
      forwardSubagentText: true,
    };

    const sdkQuery = queryFn({ ...params, prompt, options });

    async function* tracedGenerator(): AsyncGenerator<SDKMessage, void> {
      // finalize() / finalizeError() are idempotent (guarded by `ended`), so
      // calling finalize() unconditionally in `finally` is safe even when the
      // error path already called finalizeError(). The `finally` block matters
      // because consumers can exit early (`break`) and skip the post-loop code.
      try {
        for await (const msg of sdkQuery) {
          dispatch(ctx, msg);
          yield msg;
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        await ctx.finalizeError(message);
        throw err;
      } finally {
        await ctx.finalize();
      }
    }

    const wrappedIterator = tracedGenerator();

    return new Proxy(sdkQuery, {
      get(target, prop, receiver) {
        if (prop === Symbol.asyncIterator) {
          return () => wrappedIterator;
        }
        if (prop === 'next' || prop === 'return' || prop === 'throw') {
          return Reflect.get(wrappedIterator, prop, wrappedIterator);
        }
        if (prop === 'interrupt') {
          const method = Reflect.get(target, prop, receiver) as
            | ((...a: unknown[]) => unknown)
            | undefined;
          return (...args: unknown[]) => {
            // Fire-and-forget the trace flush so .interrupt() stays sync-compatible.
            void ctx.finalizeError('interrupted');
            return typeof method === 'function' ? method.call(target, ...args) : method;
          };
        }
        // eslint-disable-next-line @typescript-eslint/no-unsafe-return
        return Reflect.get(target, prop, receiver);
      },
    });
  };
}

function dispatch(ctx: LiveTracingContext, msg: SDKMessage): void {
  const m = msg as { type: string; subtype?: string };
  switch (m.type) {
    case 'system':
      if (m.subtype === 'init') {
        ctx.onSystemInit(msg as unknown as SDKSystemInit);
      }
      return;
    case 'assistant':
      ctx.onAssistantMessage(msg as unknown as SDKAssistantLike);
      return;
    case 'user':
      ctx.onUserMessage(msg as unknown as SDKUserLike);
      return;
    case 'result':
      ctx.onResultMessage(msg as unknown as SDKResultLike);
      return;
    default:
      // Ignore status, partial assistant, hook, task progress, etc. — these
      // are heartbeat signals that don't change the span tree.
      return;
  }
}

/**
 * Initial value for the root span's `prompt` input. For string prompts we have
 * it up front; for streaming prompts we start with a placeholder and append
 * each user message's text via `capturingPromptIterable`.
 */
function initialPromptFor(prompt: string | AsyncIterable<unknown>): string {
  return typeof prompt === 'string' ? prompt : '';
}

/**
 * Wrap an AsyncIterable of SDK user messages so each yielded message's text
 * content is forwarded to `onText`. Used to record streaming prompts on the
 * root span's inputs as they arrive; the underlying iterable is passed through
 * unchanged to the SDK.
 */
function capturingPromptIterable(
  source: AsyncIterable<unknown>,
  onText: (text: string) => void,
): AsyncIterable<unknown> {
  return {
    [Symbol.asyncIterator](): AsyncIterator<unknown> {
      const sourceIter = source[Symbol.asyncIterator]();
      return {
        async next(): Promise<IteratorResult<unknown>> {
          const result = await sourceIter.next();
          if (!result.done) {
            const text = extractUserText(result.value);
            if (text) {
              onText(text);
            }
          }
          return result;
        },
        return: sourceIter.return ? sourceIter.return.bind(sourceIter) : undefined,
        throw: sourceIter.throw ? sourceIter.throw.bind(sourceIter) : undefined,
      };
    },
  };
}

function extractUserText(value: unknown): string | undefined {
  if (typeof value !== 'object' || value == null) {
    return undefined;
  }
  const v = value as { type?: string; message?: { content?: unknown } };
  if (v.type !== 'user') {
    return undefined;
  }
  const content = v.message?.content;
  if (typeof content === 'string') {
    return content;
  }
  if (Array.isArray(content)) {
    const text = content
      .filter((b): b is { type: 'text'; text: string } => {
        return typeof b === 'object' && b != null && (b as { type?: string }).type === 'text';
      })
      .map((b) => b.text)
      .join('');
    return text || undefined;
  }
  return undefined;
}
