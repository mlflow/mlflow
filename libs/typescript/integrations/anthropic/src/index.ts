/**
 * Main tracedAnthropic wrapper function for MLflow tracing integration
 */

import { withSpan, LiveSpan, SpanAttributeKey, SpanType, TokenUsage } from 'mlflow-tracing';

const SUPPORTED_MODULES = ['Messages'];
const SUPPORTED_METHODS = ['create'];

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
