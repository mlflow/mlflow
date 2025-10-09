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
const SUPPORTED_METHODS = ['generateContent', 'countTokens', 'embedContent'];

/**
 * Create a traced version of Gemini client with MLflow tracing
 * @param geminiClient - The Gemini client instance to trace
 * @returns Traced Gemini client with tracing capabilities
 */
export function tracedGemini<T = any>(geminiClient: T): T {
  const tracedClient = new Proxy(geminiClient as any, {
    get(target, prop, receiver) {
      const original = Reflect.get(target, prop, receiver);
      const moduleName = (target as object).constructor?.name?.toLowerCase?.() || getModuleName(target);

      if (typeof original === 'function') {
        if (shouldTraceMethod(moduleName, String(prop))) {
          // eslint-disable-next-line @typescript-eslint/ban-types
          return wrapWithTracing(original as Function, moduleName!, String(prop));
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

function getModuleName(target: any): string | undefined {
  // Try to infer module name for Gemini SDK (e.g., "models")
  if (target && typeof target === 'object') {
    if (target === Object(target) && target.constructor && target.constructor.name) {
      return target.constructor.name.toLowerCase();
    }
    if (target === Object(target) && target.name) {
      return String(target.name).toLowerCase();
    }
  }
  return undefined;
}

function shouldTraceMethod(moduleName: string | undefined, methodName: string): boolean {
  if (!moduleName) {
    return false;
  }
  return SUPPORTED_MODULES.includes(moduleName) && SUPPORTED_METHODS.includes(methodName);
}

// eslint-disable-next-line @typescript-eslint/ban-types
function wrapWithTracing(fn: Function, moduleName: string, methodName: string): Function {
  const spanType = getSpanType(methodName);
  const name = getSpanName(methodName);

  return function (this: any, ...args: any[]) {
    if (!spanType) {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-return
      return fn.apply(this, args);
    }

    // eslint-disable-next-line @typescript-eslint/no-unsafe-return
    return withSpan(
      async (span: LiveSpan) => {
        if (args.length === 1) {
          span.setInputs(args[0]);
        } else if (args.length > 1) {
          span.setInputs(args);
        }

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

        // eslint-disable-next-line @typescript-eslint/no-unsafe-return
        return result;
      },
      { name, spanType }
    );
  };
}

function getSpanType(methodName: string): SpanType | undefined {
  switch (methodName) {
    case 'generateContent':
    case 'countTokens':
      return SpanType.LLM;
    case 'embedContent':
      return SpanType.EMBEDDING;
    default:
      return undefined;
  }
}

function getSpanName(methodName: string): string {
  switch (methodName) {
    case 'generateContent':
      return 'Gemini.generateContent';
    case 'countTokens':
      return 'Gemini.countTokens';
    case 'embedContent':
      return 'Gemini.embedContent';
    default:
      return `Gemini.${methodName}`;
  }
}

function extractTokenUsage(response: any): TokenUsage | undefined {
  const usage = response?.usageMetadata ?? response?.usage;
  if (!usage) {
    return deriveFromCountTokens(response);
  }

  const input = usage.promptTokenCount ?? usage.inputTokenCount ?? usage.inputTokens;
  const output =
    usage.candidatesTokenCount ??
    usage.outputTokenCount ??
    usage.completionTokenCount ??
    usage.outputTokens;
  const total = usage.totalTokenCount ?? usage.totalTokens;

  if (
    typeof input === 'number' &&
    typeof output === 'number' &&
    typeof total === 'number'
  ) {
    return {
      input_tokens: input,
      output_tokens: output,
      total_tokens: total
    };
  }

  return undefined;
}

function deriveFromCountTokens(response: any): TokenUsage | undefined {
  const total = response?.totalTokenCount ?? response?.totalTokens;
  if (typeof total !== 'number') {
    return undefined;
  }

  const input = response?.inputTokenCount ?? response?.inputTokens ?? total;
  const output = response?.outputTokenCount ?? response?.outputTokens ?? 0;

  return {
    input_tokens: typeof input === 'number' ? input : total,
    output_tokens: typeof output === 'number' ? output : 0,
    total_tokens: total
  };
}

function isProxyable(value: any): value is object {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}