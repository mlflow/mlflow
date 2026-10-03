/**
 * Main tracedOpenAI wrapper function for MLflow tracing integration
 */

import { CompletionUsage } from 'openai/resources/index';
import { ResponseStreamEvent, ResponseUsage } from 'openai/resources/responses/responses';
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
const MAX_ACCUMULATED_LENGTH = 10000;
const TRUNCATION_SUFFIX = '...[truncated]';

type StreamInternals<Item> = { iterator: () => AsyncIterator<Item> };
type ChatCompletionStreamState = {
  span: LiveSpan;
  choices: Map<number, AccumulatedChoice>;
  usage?: TokenUsage;
  consumed: boolean;
  ended: boolean;
};
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
type ResponsesStreamState = {
  span: LiveSpan;
  outputItems: Map<number, Record<string, unknown>>;
  outputText: string;
  completedResponse?: Record<string, unknown>;
  usage?: TokenUsage;
  consumed: boolean;
  ended: boolean;
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

    if (
      (moduleName === 'Completions' || moduleName === 'Responses') &&
      // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
      args[0]?.stream === true
    ) {
      const parentSpan = getCurrentActiveSpan();
      const span = startSpan({ name, spanType, parent: parentSpan ?? undefined });
      span.setInputs(args[0]);
      span.setAttribute(SpanAttributeKey.MESSAGE_FORMAT, 'openai');

      // eslint-disable-next-line @typescript-eslint/no-unsafe-return
      return Promise.resolve()
        .then(() => fn.apply(this, args) as unknown)
        .then((stream) =>
          moduleName === 'Responses'
            ? wrapResponsesStream(stream, span)
            : wrapChatCompletionStream(stream, span),
        )
        .catch((error: unknown) => {
          const err = error instanceof Error ? error : new Error(String(error));
          span.setStatus(SpanStatusCode.ERROR, err.message);
          span.recordException(err);
          span.end();
          throw error;
        });
    }

    // eslint-disable-next-line @typescript-eslint/no-unsafe-return
    return withSpan(
      async (span: LiveSpan) => {
        span.setInputs(args[0]);

        const result = await fn.apply(this, args);

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
  const state: ChatCompletionStreamState = {
    span,
    choices: new Map(),
    consumed: false,
    ended: false,
  };

  // A consumer may abandon the stream without ever iterating it, in which case
  // no iterator runs and the span would stay open forever. Aborting the request
  // is the one deterministic signal that this happened. Once consumption has
  // started the iterator owns the span instead: the SDK aborts the controller
  // on both early `break` and mid-stream failure, and ending the span here
  // would race the iterator and discard its status and accumulated output.
  targetStream.controller?.signal?.addEventListener(
    'abort',
    () => {
      if (!state.consumed) {
        endChatCompletionStreamSpan(state);
      }
    },
    { once: true },
  );

  // On a real SDK `Stream`, `iterator` is the single choke point for every
  // consumption path: `Symbol.asyncIterator()`, `tee()` and `toReadableStream()`
  // all call it. It is typed private, hence the cast. Anything else (a
  // duck-typed stream) is wrapped at `Symbol.asyncIterator` instead; wrapping
  // both would accumulate every chunk twice.
  const internals = targetStream as unknown as StreamInternals<ChatCompletionChunk>;
  const wrapsInternalIterator = typeof internals.iterator === 'function';

  return new Proxy(targetStream, {
    get(target, prop, receiver) {
      if (wrapsInternalIterator ? prop === 'iterator' : prop === Symbol.asyncIterator) {
        return () => {
          state.consumed = true;
          const iterator = wrapsInternalIterator
            ? internals.iterator()
            : (target as AsyncIterable<ChatCompletionChunk>)[Symbol.asyncIterator]();
          return wrapChatCompletionIterator(iterator, state);
        };
      }

      // NB: values are deliberately returned unbound. Binding them to `target`
      // would make `tee()` and `toReadableStream()` read the unwrapped
      // `iterator` and escape tracing entirely.
      return Reflect.get(target, prop, receiver) as unknown;
    },
  });
}

function wrapResponsesStream(stream: unknown, span: LiveSpan): Stream<ResponseStreamEvent> {
  const targetStream = stream as Stream<ResponseStreamEvent>;
  const state: ResponsesStreamState = {
    span,
    outputItems: new Map(),
    outputText: '',
    consumed: false,
    ended: false,
  };

  targetStream.controller?.signal?.addEventListener(
    'abort',
    () => {
      if (!state.consumed) {
        endResponsesStreamSpan(state);
      }
    },
    { once: true },
  );

  const internals = targetStream as unknown as StreamInternals<ResponseStreamEvent>;
  const wrapsInternalIterator = typeof internals.iterator === 'function';

  return new Proxy(targetStream, {
    get(target, prop, receiver) {
      if (wrapsInternalIterator ? prop === 'iterator' : prop === Symbol.asyncIterator) {
        return () => {
          state.consumed = true;
          const iterator = wrapsInternalIterator
            ? internals.iterator()
            : (target as AsyncIterable<ResponseStreamEvent>)[Symbol.asyncIterator]();
          return wrapResponsesIterator(iterator, state);
        };
      }

      return Reflect.get(target, prop, receiver) as unknown;
    },
  });
}

function recordStreamError(state: { span: LiveSpan }, error: unknown): void {
  const err = error instanceof Error ? error : new Error(String(error));
  state.span.setStatus(SpanStatusCode.ERROR, err.message);
  state.span.recordException(err);
}

function endChatCompletionStreamSpan(state: ChatCompletionStreamState): void {
  if (state.ended) {
    return;
  }
  state.ended = true;
  state.span.setOutputs({ choices: [...state.choices.values()] });
  if (state.usage) {
    state.span.setAttribute(SpanAttributeKey.TOKEN_USAGE, state.usage);
  }
  state.span.end();
}

function endResponsesStreamSpan(state: ResponsesStreamState): void {
  if (state.ended) {
    return;
  }
  state.ended = true;
  state.span.setOutputs(buildResponsesStreamOutputs(state));
  if (state.usage) {
    state.span.setAttribute(SpanAttributeKey.TOKEN_USAGE, state.usage);
  }
  state.span.end();
}

async function* wrapChatCompletionIterator(
  iterator: AsyncIterator<ChatCompletionChunk>,
  state: ChatCompletionStreamState,
): AsyncGenerator<ChatCompletionChunk> {
  const { choices } = state;
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
        state.usage = extractTokenUsage(value);
      }
      yield value;
    }
  } catch (error) {
    recordStreamError(state, error);
    throw error;
  } finally {
    if (!completed) {
      try {
        await iterator.return?.();
      } catch (error) {
        recordStreamError(state, error);
      }
    }
    endChatCompletionStreamSpan(state);
  }
}

async function* wrapResponsesIterator(
  iterator: AsyncIterator<ResponseStreamEvent>,
  state: ResponsesStreamState,
): AsyncGenerator<ResponseStreamEvent> {
  let completed = false;

  try {
    while (true) {
      const { value, done } = await iterator.next();
      if (done) {
        completed = true;
        break;
      }

      accumulateResponsesEvent(state, value);
      yield value;
    }
  } catch (error) {
    recordStreamError(state, error);
    throw error;
  } finally {
    if (!completed) {
      try {
        await iterator.return?.();
      } catch (error) {
        recordStreamError(state, error);
      }
    }
    endResponsesStreamSpan(state);
  }
}

function accumulateResponsesEvent(state: ResponsesStreamState, event: ResponseStreamEvent): void {
  switch (event.type) {
    case 'response.output_text.delta': {
      state.outputText = appendBounded(state.outputText, event.delta);
      const existing = state.outputItems.get(event.output_index) ?? {
        type: 'message',
        role: 'assistant',
        content: [{ type: 'output_text', text: '' }],
      };
      const content = Array.isArray(existing.content)
        ? [...(existing.content as Array<Record<string, unknown>>)]
        : [{ type: 'output_text', text: '' }];
      const partIndex = event.content_index ?? 0;
      const part = { ...(content[partIndex] ?? { type: 'output_text', text: '' }) };
      if (typeof part.text === 'string') {
        part.text = appendBounded(part.text, event.delta);
      } else {
        part.type = 'output_text';
        part.text = appendBounded('', event.delta);
      }
      content[partIndex] = part;
      existing.content = content;
      state.outputItems.set(event.output_index, existing);
      break;
    }
    case 'response.output_item.added':
    case 'response.output_item.done': {
      state.outputItems.set(event.output_index, boundOutputItem(event.item));
      break;
    }
    case 'response.function_call_arguments.delta': {
      const existing = state.outputItems.get(event.output_index) ?? {
        type: 'function_call',
        arguments: '',
      };
      const currentArgs = typeof existing.arguments === 'string' ? existing.arguments : '';
      existing.type = 'function_call';
      existing.arguments = appendBounded(currentArgs, event.delta);
      state.outputItems.set(event.output_index, existing);
      break;
    }
    case 'response.function_call_arguments.done': {
      const existing = state.outputItems.get(event.output_index) ?? {
        type: 'function_call',
        arguments: '',
      };
      existing.type = 'function_call';
      existing.arguments = truncateBounded(event.arguments);
      state.outputItems.set(event.output_index, existing);
      break;
    }
    case 'response.completed': {
      state.completedResponse = boundCompletedResponse(event.response);
      state.usage = extractTokenUsage(event.response);
      break;
    }
    case 'response.failed': {
      const message = event.response.error?.message ?? 'Responses stream failed';
      recordStreamError(state, new Error(message));
      break;
    }
    case 'error': {
      recordStreamError(state, new Error(event.message));
      break;
    }
    default:
      break;
  }
}

function buildResponsesStreamOutputs(state: ResponsesStreamState): unknown {
  if (state.completedResponse) {
    return state.completedResponse;
  }

  const output = [...state.outputItems.entries()]
    .sort(([left], [right]) => left - right)
    .map(([, item]) => item);

  if (output.length === 0 && state.outputText) {
    return {
      output_text: state.outputText,
      output: [
        {
          type: 'message',
          role: 'assistant',
          content: [{ type: 'output_text', text: state.outputText }],
        },
      ],
    };
  }

  return {
    ...(state.outputText ? { output_text: state.outputText } : {}),
    output,
  };
}

function boundCompletedResponse(response: unknown): Record<string, unknown> {
  const value = (response ?? {}) as Record<string, unknown>;
  const output = Array.isArray(value.output)
    ? value.output.map((item) => boundOutputItem(item))
    : [];

  return {
    id: value.id,
    object: value.object ?? 'response',
    model: value.model,
    status: value.status,
    output,
    usage: value.usage,
    ...(typeof value.output_text === 'string'
      ? { output_text: truncateBounded(value.output_text) }
      : {}),
  };
}

function boundOutputItem(item: unknown): Record<string, unknown> {
  const value = { ...((item ?? {}) as Record<string, unknown>) };

  if (typeof value.arguments === 'string') {
    value.arguments = truncateBounded(value.arguments);
  }

  if (Array.isArray(value.content)) {
    value.content = value.content.map((part) => {
      const contentPart = { ...((part ?? {}) as Record<string, unknown>) };
      if (typeof contentPart.text === 'string') {
        contentPart.text = truncateBounded(contentPart.text);
      }
      return contentPart;
    });
  }

  return value;
}

function appendBounded(value: string, delta: string): string {
  if (value.endsWith(TRUNCATION_SUFFIX)) {
    return value;
  }
  const combined = value + delta;
  if (combined.length <= MAX_ACCUMULATED_LENGTH) {
    return combined;
  }
  return combined.slice(0, MAX_ACCUMULATED_LENGTH - TRUNCATION_SUFFIX.length) + TRUNCATION_SUFFIX;
}

function truncateBounded(value: string): string {
  return appendBounded('', value);
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
