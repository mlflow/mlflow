/**
 * Main tracedAnthropic wrapper function for MLflow tracing integration
 */

import {
  startSpan,
  getCurrentActiveSpan,
  SpanAttributeKey,
  SpanType,
  SpanStatusCode,
  TokenUsage,
  LiveSpan,
  withSpan
} from 'mlflow-tracing';

const SUPPORTED_MODULES = ['Messages'];
const SUPPORTED_METHODS = ['create', 'stream'];

/**
 * Create a traced version of Anthropic client with MLflow tracing
 * @param anthropicClient - The Anthropic client instance to trace
 * @returns Traced Anthropic client with tracing capabilities
 */
export function tracedAnthropic<T = any>(anthropicClient: T): T {
  const tracedClient = new Proxy(anthropicClient as any, {
    get(target, prop, receiver) {
      const original = Reflect.get(target, prop, receiver);
      const moduleName = (target as object).constructor?.name;

      if (typeof original === 'function') {
        if (shouldTraceMethod(moduleName, String(prop))) {
          const methodName = String(prop);
          if (methodName === 'stream') {
            // eslint-disable-next-line @typescript-eslint/ban-types
            return wrapStreamWithTracing(original as Function, moduleName) as T;
          }
          // eslint-disable-next-line @typescript-eslint/ban-types
          return wrapWithTracing(original as Function, moduleName) as T;
        }
        // eslint-disable-next-line @typescript-eslint/ban-types
        return (original as Function).bind(target) as T;
      }

      if (
        original &&
        !Array.isArray(original) &&
        !(original instanceof Date) &&
        typeof original === 'object'
      ) {
        return tracedAnthropic(original) as T;
      }

      return original as T;
    },
  });

  return tracedClient as T;
}

function shouldTraceMethod(moduleName: string | undefined, methodName: string): boolean {
  if (!moduleName) {
    return false;
  }
  return SUPPORTED_MODULES.includes(moduleName) && SUPPORTED_METHODS.includes(methodName);
}

// eslint-disable-next-line @typescript-eslint/ban-types
function wrapWithTracing(fn: Function, moduleName: string): Function {
  const spanType = getSpanType(moduleName);
  const name = moduleName;

  return function (this: any, ...args: any[]) {
    if (!spanType) {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-return
      return fn.apply(this, args);
    }

    // Skip tracing for create() calls with stream: true
    // The Anthropic SDK's stream() method internally calls create() with stream: true,
    // and we don't want to create a duplicate trace - the streaming wrapper handles it
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
    if (args[0]?.stream === true) {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-return
      return fn.apply(this, args);
    }

    // eslint-disable-next-line @typescript-eslint/no-unsafe-return
    return withSpan(
      async (span: LiveSpan) => {
        span.setInputs(args[0]);

        const result = await fn.apply(this, args);

        span.setOutputs(result);

        try {
          const usage = extractTokenUsage(result);
          if (usage) {
            span.setAttribute(SpanAttributeKey.TOKEN_USAGE, usage);
          }
        } catch (error) {
          console.debug('Error extracting token usage', error);
        }

        span.setAttribute(SpanAttributeKey.MESSAGE_FORMAT, 'anthropic');

        // eslint-disable-next-line @typescript-eslint/no-unsafe-return
        return result;
      },
      { name, spanType },
    );
  };
}

function getSpanType(moduleName: string): SpanType | undefined {
  switch (moduleName) {
    case 'Messages':
      return SpanType.LLM;
    default:
      return undefined;
  }
}

// eslint-disable-next-line @typescript-eslint/ban-types
function wrapStreamWithTracing(fn: Function, moduleName: string): Function {
  const spanType = getSpanType(moduleName);
  const name = moduleName;

  return function (this: any, ...args: any[]) {
    if (!spanType) {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-return
      return fn.apply(this, args);
    }

    // Create the stream (synchronous call)
    const stream = fn.apply(this, args);

    // Return a proxy that wraps the stream with tracing
    // eslint-disable-next-line @typescript-eslint/no-unsafe-return
    return wrapMessageStream(stream, args[0], name, spanType);
  };
}

function wrapMessageStream(stream: any, inputs: any, name: string, spanType: SpanType): any {
  let traced = false;

  return new Proxy(stream, {
    get(target, prop, receiver) {
      const original = Reflect.get(target, prop, receiver);

      // Wrap finalMessage() to add tracing
      if (prop === 'finalMessage') {
        return async function () {
          if (traced) {
            // Already traced via async iteration, just return the message
            // eslint-disable-next-line @typescript-eslint/no-unsafe-return, @typescript-eslint/no-unsafe-member-access
            return target.finalMessage();
          }
          traced = true;

          // eslint-disable-next-line @typescript-eslint/no-unsafe-return
          return withSpan(
            async (span: LiveSpan) => {
              span.setInputs(inputs);

              // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
              const message = await target.finalMessage();

              span.setOutputs(message);

              try {
                const usage = extractTokenUsage(message);
                if (usage) {
                  span.setAttribute(SpanAttributeKey.TOKEN_USAGE, usage);
                }
              } catch (error) {
                console.debug('Error extracting token usage', error);
              }

              span.setAttribute(SpanAttributeKey.MESSAGE_FORMAT, 'anthropic');

              // eslint-disable-next-line @typescript-eslint/no-unsafe-return
              return message;
            },
            { name, spanType }
          );
        };
      }

      // Wrap async iterator for `for await (const event of stream)` pattern
      if (prop === Symbol.asyncIterator) {
        return function () {
          traced = true;
          // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
          return wrapAsyncIterator(target[Symbol.asyncIterator](), target, inputs, name, spanType);
        };
      }

      if (typeof original === 'function') {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-return, @typescript-eslint/no-unsafe-member-access
        return original.bind(target);
      }
      // eslint-disable-next-line @typescript-eslint/no-unsafe-return
      return original;
    }
  });
}

async function* wrapAsyncIterator(
  iterator: AsyncIterator<any>,
  stream: any,
  inputs: any,
  name: string,
  spanType: SpanType
): AsyncGenerator<any> {
  // Use startSpan for manual lifecycle management since withSpan doesn't support async generators
  const parentSpan = getCurrentActiveSpan();
  const span = startSpan({ name, spanType, parent: parentSpan ?? undefined });
  span.setInputs(inputs);

  try {
    while (true) {
      const { value, done } = await iterator.next();
      if (done) break;
      yield value;
    }

    // After iteration completes, get the final message for outputs and token usage
    try {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
      const finalMessage = await stream.finalMessage();
      span.setOutputs(finalMessage);

      const usage = extractTokenUsage(finalMessage);
      if (usage) {
        span.setAttribute(SpanAttributeKey.TOKEN_USAGE, usage);
      }
    } catch (e) {
      // Stream may have completed without finalMessage available
      console.debug('Could not get final message from stream', e);
    }

    span.setAttribute(SpanAttributeKey.MESSAGE_FORMAT, 'anthropic');
    span.end();
  } catch (error) {
    span.setStatus(SpanStatusCode.ERROR, (error as Error).message);
    span.end();
    throw error;
  }
}

function extractTokenUsage(response: any): TokenUsage | undefined {
  // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
  const usage = response?.usage;
  if (!usage) {
    return undefined;
  }

  // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
  const inputTokens = usage.input_tokens;
  // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
  const outputTokens = usage.output_tokens;
  // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
  const totalTokens = usage.total_tokens ?? (inputTokens ?? 0) + (outputTokens ?? 0);

  return {
    input_tokens: inputTokens ?? 0,
    output_tokens: outputTokens ?? 0,
    total_tokens: totalTokens ?? 0,
  };
}
