/**
 * Main tracedOpenAI wrapper function for MLflow tracing integration
 */

import { CompletionUsage } from 'openai/resources/index';
import { ResponseUsage } from 'openai/resources/responses/responses';
import { ChatCompletionChunk } from 'openai/resources/chat/completions';
import { Stream } from 'openai/streaming';
import {
  getCurrentActiveSpan,
  startSpan,
  withSpan,
  LiveSpan,
  SpanAttributeKey,
  SpanStatusCode,
  SpanType,
  TokenUsage,
} from '@mlflow/core';

// NB: 'Completions' represents chat.completions
const SUPPORTED_MODULES = ['Completions', 'Responses', 'Embeddings'];
const SUPPORTED_METHODS = ['create']; // chat.completions.create, embeddings.create, responses.create

type OpenAIUsage = CompletionUsage | ResponseUsage;
type AccumulatedChoice = {
  index: number;
  message: {
    role: string;
    content: string;
    tool_calls?: Array<{
      id?: string;
      type: string;
      function: { name?: string; arguments: string };
    }>;
  };
  finish_reason?: string | null;
};

/**
 * Create a traced version of OpenAI client with MLflow tracing
 * @param openaiClient - The OpenAI client instance to trace
 * @param config - Optional configuration for tracing
 * @returns Traced OpenAI client with tracing capabilities
 *
 * @example
 * const openai = new OpenAI({ apiKey: 'test-key' });
 * const wrappedOpenAI = tracedOpenAI(openai);
 *
 * const response = await wrappedOpenAI.chat.completions.create({
 *   messages: [{ role: 'user', content: 'Hello!' }],
 *   model: 'gpt-4o-mini',
 *   temperature: 0.5
 * });
 *
 * // The trace for the LLM call will be logged to MLflow
 *
 */
export function tracedOpenAI<T = any>(openaiClient: T): T {
  /**
   * Create a proxy to intercept method calls
   */
  const tracedClient = new Proxy(openaiClient as any, {
    get(target, prop, receiver) {
      const original = Reflect.get(target, prop, receiver) as unknown;
      const moduleName = (target as object).constructor.name;

      if (typeof original === 'function') {
        // If reach to the end function to be traced, wrap it with tracing
        if (shouldTraceMethod(moduleName, String(prop))) {
          // eslint-disable-next-line @typescript-eslint/ban-types
          return wrapWithTracing(original, moduleName) as T;
        }
        // eslint-disable-next-line @typescript-eslint/ban-types
        return original.bind(target) as T;
      }

      // For nested objects (like chat.completions), recursively apply tracking
      if (
        original &&
        !Array.isArray(original) &&
        !(original instanceof Date) &&
        typeof original === 'object'
      ) {
        return tracedOpenAI(original) as T;
      }

      return original as T;
    },
  });
  return tracedClient as T;
}

/**
 * Determine if a method should be traced based on the target object and property
 */
function shouldTraceMethod(module: string, methodName: string): boolean {
  return SUPPORTED_MODULES.includes(module) && SUPPORTED_METHODS.includes(methodName);
}

/**
 * Wrap a function with tracing using the full method path
 *
 * @param fn - The function to wrap
 * @param target - The target module that contains the function to wrap
 * @returns The wrapped function
 */
// eslint-disable-next-line @typescript-eslint/ban-types
function wrapWithTracing(fn: Function, moduleName: string): Function {
  // Use the full method path for span type determination
  const spanType = getSpanType(moduleName);
  const name = moduleName;

  return function (this: any, ...args: any[]) {
    // If the method is not supported, return the original function
    if (!spanType) {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-return
      return fn.apply(this, args);
    }

    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
    if (moduleName === 'Completions' && args[0]?.stream === true) {
      const parentSpan = getCurrentActiveSpan();
      const span = startSpan({ name, spanType, parent: parentSpan ?? undefined });
      span.setInputs(args[0]);
      span.setAttribute(SpanAttributeKey.MESSAGE_FORMAT, 'openai');

      // eslint-disable-next-line @typescript-eslint/no-unsafe-return
      return Promise.resolve()
        .then(() => fn.apply(this, args) as unknown)
        .then((stream) => wrapChatCompletionStream(stream, span))
        .catch((error: unknown) => {
          span.setStatus(
            SpanStatusCode.ERROR,
            error instanceof Error ? error.message : String(error),
          );
          span.end();
          throw error;
        });
    }

    // eslint-disable-next-line @typescript-eslint/no-unsafe-return
    return withSpan(
      async (span: LiveSpan) => {
        span.setInputs(args[0]);

        const result = await fn.apply(this, args);

        // TODO: Handle Responses API streaming responses
        span.setOutputs(result);

        // Add token usage
        try {
          const usage = extractTokenUsage(result);
          if (usage) {
            span.setAttribute(SpanAttributeKey.TOKEN_USAGE, usage);
          }
        } catch (error) {
          console.debug('Error extracting token usage', error);
        }

        span.setAttribute(SpanAttributeKey.MESSAGE_FORMAT, 'openai');

        // eslint-disable-next-line @typescript-eslint/no-unsafe-return
        return result;
      },
      { name, spanType },
    );
  };
}

function wrapChatCompletionStream(stream: unknown, span: LiveSpan): Stream<ChatCompletionChunk> {
  const targetStream = stream as Stream<ChatCompletionChunk>;
  return new Proxy(targetStream, {
    get(target, prop, receiver) {
      if (prop === Symbol.asyncIterator) {
        return () => wrapChatCompletionIterator(target[Symbol.asyncIterator](), span);
      }

      const original = Reflect.get(target, prop, receiver) as unknown;
      if (typeof original === 'function') {
        return original.bind(target) as (...args: unknown[]) => unknown;
      }
      return original;
    },
  });
}

async function* wrapChatCompletionIterator(
  iterator: AsyncIterator<ChatCompletionChunk>,
  span: LiveSpan,
): AsyncGenerator<ChatCompletionChunk> {
  const choices = new Map<number, AccumulatedChoice>();
  let usage: TokenUsage | undefined;
  let completed = false;

  try {
    while (true) {
      const { value, done } = await iterator.next();
      if (done) {
        completed = true;
        break;
      }

      for (const choice of value.choices ?? []) {
        const accumulated = choices.get(choice.index) ?? {
          index: choice.index,
          message: { role: 'assistant', content: '' },
        };
        const delta = choice.delta ?? {};
        if (delta.role) {
          accumulated.message.role = delta.role;
        }
        if (delta.content) {
          accumulated.message.content = appendBounded(accumulated.message.content, delta.content);
        }
        if (delta.tool_calls) {
          accumulated.message.tool_calls ??= [];
          for (const toolCall of delta.tool_calls) {
            const tool = accumulated.message.tool_calls[toolCall.index] ?? {
              type: 'function',
              function: { arguments: '' },
            };
            if (toolCall.id) {
              tool.id = toolCall.id;
            }
            if (toolCall.type) {
              tool.type = toolCall.type;
            }
            if (toolCall.function?.name) {
              tool.function.name = toolCall.function.name;
            }
            if (toolCall.function?.arguments) {
              tool.function.arguments = appendBounded(
                tool.function.arguments,
                toolCall.function.arguments,
              );
            }
            accumulated.message.tool_calls[toolCall.index] = tool;
          }
        }
        if (choice.finish_reason) {
          accumulated.finish_reason = choice.finish_reason;
        }
        choices.set(choice.index, accumulated);
      }

      if (value.usage) {
        usage = extractTokenUsage(value);
      }
      yield value;
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    span.setStatus(SpanStatusCode.ERROR, message);
    throw error;
  } finally {
    if (!completed) {
      try {
        await iterator.return?.();
      } catch (error) {
        span.setStatus(
          SpanStatusCode.ERROR,
          error instanceof Error ? error.message : String(error),
        );
      }
    }
    span.setOutputs({ choices: [...choices.values()] });
    if (usage) {
      span.setAttribute(SpanAttributeKey.TOKEN_USAGE, usage);
    }
    span.end();
  }
}

function appendBounded(value: string, delta: string): string {
  const maxLength = 10000;
  return (value + delta).slice(0, maxLength);
}

/**
 * Determine span type based on the full method path
 */
function getSpanType(moduleName: string): SpanType | undefined {
  switch (moduleName) {
    case 'Completions':
      return SpanType.LLM;
    case 'Responses':
      return SpanType.LLM;
    case 'Embeddings':
      return SpanType.EMBEDDING;
    // TODO: Support other methods in the future.
    default:
      return undefined;
  }
}

/**
 * Extract token usage information from OpenAI response
 * Supports both ChatCompletion API format and Responses API format
 */
function extractTokenUsage(response: any): TokenUsage | undefined {
  // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
  const usage = response?.usage as OpenAIUsage | undefined;

  if (!usage) {
    return undefined;
  }

  // Try Responses API format first (input_tokens, output_tokens)
  if ('input_tokens' in usage) {
    return {
      input_tokens: usage.input_tokens,
      output_tokens: usage.output_tokens,
      total_tokens: usage.total_tokens || usage.input_tokens + usage.output_tokens,
    };
  }

  // Fall back to ChatCompletion API format (prompt_tokens, completion_tokens)
  if ('prompt_tokens' in usage) {
    return {
      input_tokens: usage.prompt_tokens,
      output_tokens: usage.completion_tokens ?? 0,
      total_tokens: usage.total_tokens || usage.prompt_tokens + (usage.completion_tokens ?? 0),
    };
  }

  return undefined;
}
