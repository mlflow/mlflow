/**
 * Main tracedOpenAI wrapper function for MLflow tracing integration
 */

import { CompletionUsage } from 'openai/resources/index';
import { ResponseUsage } from 'openai/resources/responses/responses';
import { withSpan, LiveSpan, SpanAttributeKey, SpanType, TokenUsage } from 'mlflow-tracing';

// NB: 'Completions' represents chat.completions
const SUPPORTED_MODULES = ['Completions', 'Responses', 'Embeddings'];
const SUPPORTED_METHODS = ['create']; // chat.completions.create, embeddings.create, responses.create

type OpenAIUsage = CompletionUsage | ResponseUsage;

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
      const original = Reflect.get(target, prop, receiver);
      const moduleName = (target as object).constructor.name;

      if (typeof original === 'function') {
        // If reach to the end function to be traced, wrap it with tracing
        if (shouldTraceMethod(moduleName, String(prop))) {
          // eslint-disable-next-line @typescript-eslint/ban-types
          return wrapWithTracing(original as Function, moduleName) as T;
        }
        // eslint-disable-next-line @typescript-eslint/ban-types
        return (original as Function).bind(target) as T;
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
    }
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

    // eslint-disable-next-line @typescript-eslint/no-unsafe-return
    return withSpan(
      async (span: LiveSpan) => {
        span.setInputs(args[0]);

        const result = await fn.apply(this, args);

        // TODO: Handle streaming responses
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
      { name, spanType }
    );
  };
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
      total_tokens: usage.total_tokens || usage.input_tokens + usage.output_tokens
    };
  }

  // Fall back to ChatCompletion API format (prompt_tokens, completion_tokens)
  if ('prompt_tokens' in usage) {
    return {
      input_tokens: usage.prompt_tokens,
      output_tokens: usage.completion_tokens,
      total_tokens: usage.total_tokens || usage.prompt_tokens + usage.completion_tokens
    };
  }

  return undefined;
}
