/**
 * MLflow Tracing wrapper for the @google/genai Gemini SDK.
 */

import {
  withSpan,
  SpanAttributeKey,
  SpanType,
  type TokenUsage,
  type LiveSpan
} from 'mlflow-tracing';

const SUPPORTED_MODULES = ['models'];
const SUPPORTED_METHODS = ['generateContent'];

/**
 * Create a traced version of Gemini client with MLflow tracing
 * @param geminiClient - The Gemini client instance to trace
 * @returns Traced Gemini client with tracing capabilities
 */
export function tracedGemini<T = any>(geminiClient: T): T {
  const tracedClient = new Proxy(geminiClient as any, {
    get(target, prop, receiver) {
      const original = Reflect.get(target, prop, receiver);
      const moduleName = (target as object).constructor?.name;

      if (typeof original === 'function') {
        if (shouldTraceMethod(moduleName, String(prop))) {
          // eslint-disable-next-line @typescript-eslint/ban-types
          return wrapWithTracing(original as Function, String(prop));
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
        return tracedGemini(original) as T;
      }

      return original as T;
    }
  });

  return tracedClient as T;
}

function shouldTraceMethod(moduleName: string | undefined, methodName: string): boolean {
  if (!moduleName) {
    return false;
  }
  const lowerModuleName = moduleName.toLowerCase();
  return SUPPORTED_MODULES.includes(lowerModuleName) && SUPPORTED_METHODS.includes(methodName);
}

// eslint-disable-next-line @typescript-eslint/ban-types
function wrapWithTracing(fn: Function, methodName: string): Function {
  const spanType = getSpanType(methodName);

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
          // eslint-disable-next-line no-console
          console.debug('Error extracting token usage', error);
        }

        span.setAttribute(SpanAttributeKey.MESSAGE_FORMAT, 'gemini');

        // eslint-disable-next-line @typescript-eslint/no-unsafe-return
        return result;
      },
      { name: methodName, spanType }
    );
  };
}

function getSpanType(methodName: string): SpanType | undefined {
  switch (methodName) {
    case 'generateContent':
      return SpanType.LLM;
    default:
      return undefined;
  }
}
function extractTokenUsage(response: any): TokenUsage | undefined {
  // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
  const usage = response?.usageMetadata ?? response?.usage;
  if (!usage) {
    return undefined;
  }
  // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
  const input = usage.promptTokenCount;
  // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
  const output = usage.candidatesTokenCount;
  // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
  const total = usage.totalTokenCount;

  if (input !== undefined && output !== undefined && total !== undefined) {
    return {
      input_tokens: input,
      output_tokens: output,
      total_tokens: total
    };
  }

  return undefined;
}
