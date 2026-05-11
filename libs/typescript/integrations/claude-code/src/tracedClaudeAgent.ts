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
    const promptForSpan = typeof params.prompt === 'string' ? params.prompt : '[streaming input]';
    const ctx = new LiveTracingContext(promptForSpan, params.options);

    // Default forwardSubagentText to true so sub-agent inner messages flow
    // through the parent stream. Without it, the SDK only emits tool_use and
    // tool_result blocks from sub-agents (heartbeat), which strips LLM and
    // nested tool spans from the resulting trace.
    const options: QueryOptions = {
      forwardSubagentText: true,
      ...params.options,
    };

    const sdkQuery = queryFn({ ...params, options });

    async function* tracedGenerator(): AsyncGenerator<SDKMessage, void> {
      try {
        for await (const msg of sdkQuery) {
          dispatch(ctx, msg);
          yield msg;
        }
        ctx.finalize();
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        ctx.finalizeError(message);
        throw err;
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
            ctx.finalizeError('interrupted');
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
