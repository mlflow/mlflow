/**
 * MLflow Tracing wrapper for LangChain BaseChatModel.
 *
 * Wraps LangChain chat models to produce well-formatted MLflow spans
 * for invoke() and stream() calls. Works with all LangChain providers:
 * ChatAnthropic, ChatOpenAI, ChatXAI, and any BaseChatModel subclass.
 */

import {
  withSpan,
  startSpan,
  getCurrentActiveSpan,
  SpanAttributeKey,
  SpanType,
  SpanStatusCode,
  type TokenUsage,
  type LiveSpan,
} from '@mlflow/core';

/**
 * Content block types that the MLflow UI's Anthropic chat normalizer supports.
 * Any other types (e.g. `redacted_thinking`) cause the normalizer to reject
 * the entire message, which prevents the Chat tab from rendering.
 */
const MLFLOW_UI_SUPPORTED_CONTENT_TYPES = new Set([
  'text',
  'image',
  'tool_use',
  'tool_result',
  'thinking',
]);

/**
 * Known LangChain model class names mapped to MLflow message format strings.
 * Used to set the MESSAGE_FORMAT span attribute for proper UI rendering.
 */
const MODEL_CLASS_TO_FORMAT: Record<string, string> = {
  ChatAnthropic: 'anthropic',
  ChatOpenAI: 'openai',
  ChatXAI: 'openai',
  ChatGoogleGenerativeAI: 'gemini',
};

/**
 * Detect the message format from the model's class name.
 * For RunnableBinding (result of bindTools/withConfig), checks the inner bound model.
 * Falls back to 'langchain' for unknown model types.
 */
function detectMessageFormat(model: any): string {
  const target = model?.bound ?? model;
  const className = target?.constructor?.name;
  if (className && className in MODEL_CLASS_TO_FORMAT) {
    return MODEL_CLASS_TO_FORMAT[className];
  }
  return 'langchain';
}

/**
 * Extract model configuration metadata from a LangChain model instance.
 * Handles both direct model instances and RunnableBinding (from bindTools).
 */
function extractModelConfig(model: any): Record<string, unknown> {
  const config: Record<string, unknown> = {};

  // For RunnableBinding (result of bindTools), reach into the bound model
  const baseModel = model?.bound ?? model;

  // Model name
  if (baseModel.model) {
    config.model = baseModel.model;
  } else if (baseModel.modelName) {
    config.model = baseModel.modelName;
  }

  // Common hyperparameters
  if (baseModel.temperature != null) {
    config.temperature = baseModel.temperature;
  }
  if (baseModel.maxTokens != null) {
    config.max_tokens = baseModel.maxTokens;
  }
  if (baseModel.topP != null) {
    config.top_p = baseModel.topP;
  }
  if (baseModel.topK != null) {
    config.top_k = baseModel.topK;
  }
  if (baseModel.frequencyPenalty != null) {
    config.frequency_penalty = baseModel.frequencyPenalty;
  }
  if (baseModel.presencePenalty != null) {
    config.presence_penalty = baseModel.presencePenalty;
  }

  // Stop sequences
  const stop = baseModel.stopSequences ?? baseModel.stop;
  if (stop && Array.isArray(stop) && stop.length > 0) {
    config.stop = stop;
  }

  // Tools and tool_choice from bound model configuration.
  // Different LangChain providers store these in different places:
  // - Anthropic: RunnableBinding.config (via base Runnable.withConfig)
  // - OpenAI: ChatOpenAI.defaultOptions (OpenAI overrides withConfig to return a new instance)
  // - Legacy: RunnableBinding.kwargs (deprecated, kept for compatibility)
  const bindingTools = model?.config?.tools ?? model?.defaultOptions?.tools ?? model?.kwargs?.tools;
  if (bindingTools && Array.isArray(bindingTools)) {
    config.tools = bindingTools;
  }
  const bindingToolChoice = model?.config?.tool_choice ?? model?.defaultOptions?.tool_choice ?? model?.kwargs?.tool_choice;
  if (bindingToolChoice != null) {
    config.tool_choice = bindingToolChoice;
  }

  return config;
}

/**
 * Options for configuring tracedModel behavior.
 */
export interface TracedModelOptions {
  /**
   * Explicit message format to use for span serialization.
   * Bypasses auto-detection from the model's class name.
   * Useful when class names are mangled by bundlers/minifiers.
   * Valid values: 'anthropic', 'openai', 'gemini', 'langchain'
   */
  messageFormat?: string;
}

/**
 * Create a traced version of a LangChain BaseChatModel with MLflow tracing.
 *
 * Wraps `invoke()` and `stream()` to produce LLM spans with:
 * - Input messages, model name, hyperparameters, and tools
 * - Output content with token usage statistics
 * - Token usage from usage_metadata
 * - Message format (auto-detected from model class name, or explicit via options)
 *
 * Also wraps `bindTools()` so that models with bound tools remain traced.
 *
 * @param model - The LangChain BaseChatModel instance to trace
 * @param options - Optional configuration (e.g. explicit messageFormat)
 * @returns Traced model with the same interface
 */
export function tracedModel<T = any>(model: T, options?: TracedModelOptions): T {
  if (!model || typeof model !== 'object') {
    return model;
  }

  const explicitFormat = options?.messageFormat;

  return new Proxy(model as any, {
    get(target, prop, receiver) {
      const original = Reflect.get(target, prop, receiver);

      if (typeof original === 'function') {
        if (prop === 'invoke') {
          return wrapInvoke(original as Function, target, explicitFormat);
        }
        if (prop === 'stream') {
          return wrapStream(original as Function, target, explicitFormat);
        }
        if (prop === 'bindTools') {
          return function (this: any, ...args: any[]) {
            const bound = original.apply(target, args);
            return tracedModel(bound, options);
          };
        }
        return original.bind(target);
      }

      return original;
    },
  }) as T;
}

/**
 * Extract call-time options (tools, tool_choice) from invoke/stream's second argument.
 * These override or supplement the model-level config.
 */
function extractCallOptions(callOptions: any): Record<string, unknown> {
  if (!callOptions || typeof callOptions !== 'object') {return {};}
  const opts: Record<string, unknown> = {};
  if (callOptions.tools && Array.isArray(callOptions.tools)) {
    opts.tools = callOptions.tools;
  }
  if (callOptions.tool_choice != null) {
    opts.tool_choice = callOptions.tool_choice;
  }
  return opts;
}

/**
 * Wrap the invoke() method with MLflow tracing.
 * Creates an LLM span that captures inputs, outputs, and token usage.
 * Model config is captured at call time to reflect the current state.
 */
function wrapInvoke(fn: Function, target: any, explicitFormat: string | undefined): Function {
  return function (this: any, ...args: any[]) {
    return withSpan(
      async (span: LiveSpan) => {
        const messageFormat = explicitFormat ?? detectMessageFormat(target);
        const modelConfig = extractModelConfig(target);
        const callOpts = extractCallOptions(args[1]);
        const mergedConfig = { ...modelConfig, ...callOpts };
        span.setInputs(serializeInput(args[0], messageFormat, mergedConfig));

        const result = await fn.apply(target, args);

        const usage = extractTokenUsage(result);

        span.setOutputs(serializeOutput(result, messageFormat, usage));

        try {
          if (usage) {
            span.setAttribute(SpanAttributeKey.TOKEN_USAGE, usage);
          }
        } catch (error) {
          console.debug('Error extracting token usage from LangChain response', error);
        }

        span.setAttribute(SpanAttributeKey.MESSAGE_FORMAT, messageFormat);

        return result;
      },
      { name: 'ChatModel', spanType: SpanType.LLM },
    );
  };
}

/**
 * Wrap the stream() method with MLflow tracing.
 * Creates an LLM span that wraps the async iterator, collecting chunks
 * and recording the aggregated result when iteration completes.
 * Model config is captured at call time to reflect the current state.
 */
function wrapStream(fn: Function, target: any, explicitFormat: string | undefined): Function {
  return function (this: any, ...args: any[]) {
    const messageFormat = explicitFormat ?? detectMessageFormat(target);
    const modelConfig = extractModelConfig(target);
    const callOpts = extractCallOptions(args[1]);
    const mergedConfig = { ...modelConfig, ...callOpts };
    const inputs = serializeInput(args[0], messageFormat, mergedConfig);

    // stream() returns an IterableReadableStream (async iterable).
    // We need to wrap the async iterator to trace the full lifecycle.
    let streamPromise;
    try {
      streamPromise = fn.apply(target, args);
    } catch (error) {
      // stream() threw synchronously — record an error span
      const parentSpan = getCurrentActiveSpan();
      const span = startSpan({ name: 'ChatModel', spanType: SpanType.LLM, parent: parentSpan ?? undefined });
      span.setInputs(inputs);
      span.setAttribute(SpanAttributeKey.MESSAGE_FORMAT, messageFormat);
      span.setStatus(SpanStatusCode.ERROR, (error as Error).message);
      span.end();
      throw error;
    }

    // LangChain's stream() can be sync or async depending on version.
    // Handle both cases by always wrapping in a promise.
    if (streamPromise && typeof streamPromise.then === 'function') {
      // Async: stream() returns a Promise<IterableReadableStream>
      return (streamPromise as Promise<any>).then(
        (stream: any) => wrapAsyncIterable(stream, inputs, messageFormat),
        (error: Error) => {
          // stream() promise rejected — record an error span
          const parentSpan = getCurrentActiveSpan();
          const span = startSpan({ name: 'ChatModel', spanType: SpanType.LLM, parent: parentSpan ?? undefined });
          span.setInputs(inputs);
          span.setAttribute(SpanAttributeKey.MESSAGE_FORMAT, messageFormat);
          span.setStatus(SpanStatusCode.ERROR, error.message);
          span.end();
          throw error;
        },
      );
    }

    // Sync: stream() returns an IterableReadableStream directly
    return wrapAsyncIterable(streamPromise, inputs, messageFormat);
  };
}

/**
 * Wrap an async iterable (stream result) with MLflow tracing.
 * Returns a proxy that intercepts Symbol.asyncIterator to add span tracking.
 */
function wrapAsyncIterable(stream: any, inputs: any, messageFormat: string): any {
  let tracingClaimed = false;

  return new Proxy(stream, {
    get(target, prop, receiver) {
      const original = Reflect.get(target, prop, receiver);

      if (prop === Symbol.asyncIterator) {
        return function () {
          if (tracingClaimed) {
            return target[Symbol.asyncIterator]();
          }
          tracingClaimed = true;
          return wrapStreamIterator(target[Symbol.asyncIterator](), inputs, messageFormat);
        };
      }

      if (typeof original === 'function') {
        return original.bind(target);
      }
      return original;
    },
  });
}

/**
 * Wrap an async iterator with MLflow span tracking.
 * Collects all chunks during iteration and records aggregated outputs on completion.
 */
async function* wrapStreamIterator(
  iterator: AsyncIterator<any>,
  inputs: any,
  messageFormat: string,
): AsyncGenerator<any> {
  const parentSpan = getCurrentActiveSpan();
  const span = startSpan({ name: 'ChatModel', spanType: SpanType.LLM, parent: parentSpan ?? undefined });
  span.setInputs(inputs);

  const chunks: any[] = [];
  let iterationError: Error | undefined;

  try {
    while (true) {
      const { value, done } = await iterator.next();
      if (done) {
        break;
      }
      chunks.push(value);
      yield value;
    }
  } catch (error) {
    iterationError = error as Error;
    throw error;
  } finally {
    if (iterationError) {
      span.setAttribute(SpanAttributeKey.MESSAGE_FORMAT, messageFormat);
      span.setStatus(SpanStatusCode.ERROR, iterationError.message);
      span.end();
    } else {
      try {
        // Aggregate chunks using LangChain's concat pattern
        const aggregated = aggregateChunks(chunks);
        if (aggregated) {
          const usage = extractTokenUsage(aggregated);

          span.setOutputs(serializeOutput(aggregated, messageFormat, usage));

          if (usage) {
            span.setAttribute(SpanAttributeKey.TOKEN_USAGE, usage);
          }
        }
      } catch (e) {
        console.debug('Could not aggregate stream chunks', e);
      }

      span.setAttribute(SpanAttributeKey.MESSAGE_FORMAT, messageFormat);
      span.end();
    }
  }
}

/**
 * Aggregate LangChain AIMessageChunks using their concat() method.
 * This is the same pattern used by LangChain internally for combining streamed chunks.
 */
function aggregateChunks(chunks: any[]): any {
  if (chunks.length === 0) {return undefined;}
  if (chunks.length === 1) {return chunks[0];}

  try {
    // LangChain AIMessageChunk implements concat()
    return chunks.slice(1).reduce((acc: any, chunk: any) => acc.concat(chunk), chunks[0]);
  } catch {
    // Fallback: return last chunk if concat not available
    return chunks[chunks.length - 1];
  }
}

/**
 * Map LangChain message type to a standard role string.
 * LangChain uses _getType() returning "human", "ai", etc.
 */
function langchainTypeToRole(msg: any): string {
  const type: string = msg._getType?.() ?? msg.type ?? msg.constructor?.name ?? 'unknown';
  switch (type) {
    case 'human':
      return 'user';
    case 'ai':
      return 'assistant';
    case 'system':
      return 'system';
    case 'tool':
      return 'tool';
    case 'function':
      return 'function';
    default:
      return type;
  }
}

/**
 * Convert LangChain tool_calls to OpenAI tool_calls format.
 * LangChain: {name, args, id}  →  OpenAI: {id, type: "function", function: {name, arguments}}
 */
function langchainToolCallsToOpenAI(toolCalls: any[]): any[] {
  return toolCalls.map((tc: any) => ({
    id: tc.id ?? '',
    type: 'function',
    function: {
      name: tc.name,
      arguments: typeof tc.args === 'string' ? tc.args : JSON.stringify(tc.args ?? {}),
    },
  }));
}

/**
 * Serialize a single LangChain message for the langchain format.
 * Produces {content, type, additional_kwargs, response_metadata, ...}
 */
function serializeLangchainMessage(msg: any): any {
  const type: string = msg._getType?.() ?? msg.type ?? 'unknown';
  const result: Record<string, unknown> = {
    content: msg.content,
    type,
    additional_kwargs: msg.additional_kwargs ?? {},
    response_metadata: msg.response_metadata ?? {},
  };
  if (msg.tool_calls && msg.tool_calls.length > 0) {
    result.tool_calls = msg.tool_calls;
  }
  if (msg.tool_call_id) {
    result.tool_call_id = msg.tool_call_id;
  }
  if (msg.name) {
    result.name = msg.name;
  }
  if (msg.id) {
    result.id = msg.id;
  }
  return result;
}

/**
 * Serialize LangChain messages to Anthropic API format for span recording.
 * The Anthropic API uses a separate `system` parameter (not in messages array),
 * and only accepts `user` and `assistant` roles in messages. LangChain tool
 * messages are converted to `user` messages with `tool_result` content blocks.
 */
function serializeAnthropicInput(input: any[], modelConfig: Record<string, unknown>): any {
  const systemParts: string[] = [];
  const messages: Record<string, unknown>[] = [];

  for (const msg of input) {
    const role = langchainTypeToRole(msg);
    const content = msg.content;

    if (role === 'system') {
      // Anthropic API sends system messages as a top-level parameter
      if (typeof content === 'string') {
        systemParts.push(content);
      }
      continue;
    }

    if (role === 'tool') {
      // Anthropic API represents tool results as user messages with tool_result content blocks
      messages.push({
        role: 'user',
        content: [{
          type: 'tool_result',
          tool_use_id: msg.tool_call_id ?? '',
          content: typeof content === 'string' ? content : JSON.stringify(content),
        }],
      });
      continue;
    }

    const result: Record<string, unknown> = { role };

    if (msg.tool_calls && msg.tool_calls.length > 0) {
      // Assistant messages with tool calls use content blocks
      const contentBlocks: any[] = [];
      if (typeof content === 'string' && content) {
        contentBlocks.push({ type: 'text', text: content });
      } else if (Array.isArray(content)) {
        // When thinking is enabled, LangChain returns content as an array
        // of blocks (thinking, text, etc.) alongside tool_calls
        for (const block of content) {
          if (typeof block === 'string') {
            if (block) {contentBlocks.push({ type: 'text', text: block });}
          } else if (block.type && MLFLOW_UI_SUPPORTED_CONTENT_TYPES.has(block.type as string)) {
            contentBlocks.push(block);
          } else if (block.type) {
            contentBlocks.push({ type: 'text', text: `[${block.type as string}]` });
          }
        }
      }
      for (const tc of msg.tool_calls as any[]) {
        contentBlocks.push({
          type: 'tool_use',
          id: tc.id ?? '',
          name: tc.name,
          input: tc.args ?? {},
        });
      }
      result.content = contentBlocks;
    } else if (Array.isArray(content)) {
      // Sanitize array content blocks to replace unsupported types
      result.content = (content).map((block: any) => {
        if (typeof block === 'string') {return { type: 'text', text: block };}
        if (block.type && MLFLOW_UI_SUPPORTED_CONTENT_TYPES.has(block.type as string)) {
          return block;
        }
        if (block.type) {
          return { type: 'text', text: `[${block.type as string}]` };
        }
        return { type: 'text', text: block.text ?? String(block) };
      });
    } else {
      result.content = content;
    }

    messages.push(result);
  }

  const output: Record<string, unknown> = { messages, ...modelConfig };
  if (systemParts.length > 0) {
    output.system = systemParts.join('\n');
  }
  return output;
}

/**
 * Serialize LangChain message input for span recording.
 * Produces format-specific structures that the MLflow UI can parse
 * to render the Chat tab in the trace/span explorer.
 * Includes model name, hyperparameters, and tool definitions alongside messages.
 */
function serializeInput(input: any, messageFormat: string, modelConfig: Record<string, unknown>): any {
  if (!input) {return input;}

  if (!Array.isArray(input)) {
    return input;
  }

  if (messageFormat === 'langchain') {
    // LangChain format: [{content, type: "human"/"ai"/..., additional_kwargs, ...}]
    // Include model config as a wrapper object
    const serializedMessages = input.map((msg: any) => {
      if (msg && typeof msg === 'object' && msg.content !== undefined) {
        return serializeLangchainMessage(msg);
      }
      return msg;
    });
    if (Object.keys(modelConfig).length > 0) {
      return { messages: serializedMessages, ...modelConfig };
    }
    return serializedMessages;
  }

  if (messageFormat === 'gemini') {
    // Gemini format: {contents: [{role: "user"/"model", parts: [{text: "..."}]}]}
    const contents = input.map((msg: any) => {
      const role = langchainTypeToRole(msg);
      const content = msg.content;
      const parts = typeof content === 'string'
        ? [{ text: content }]
        : Array.isArray(content) ? content.map((p: any) => ({ text: p.text ?? String(p) })) : [{ text: String(content) }];
      return { role: role === 'assistant' ? 'model' : role, parts };
    });
    return { contents, ...modelConfig };
  }

  if (messageFormat === 'anthropic') {
    return serializeAnthropicInput(input, modelConfig);
  }

  // OpenAI format: {messages: [{role, content, tool_calls?}]}
  const messages = input.map((msg: any) => {
    if (msg && typeof msg === 'object' && msg.content !== undefined) {
      const role = langchainTypeToRole(msg);
      const result: Record<string, unknown> = {
        role,
        content: msg.content,
      };
      if (msg.name) {
        result.name = msg.name;
      }
      if (msg.tool_calls && msg.tool_calls.length > 0) {
        result.tool_calls = langchainToolCallsToOpenAI(msg.tool_calls);
      }
      if (msg.tool_call_id) {
        result.tool_call_id = msg.tool_call_id;
      }
      return result;
    }
    return msg;
  });
  return { messages, ...modelConfig };
}

/**
 * Serialize LangChain message output for span recording.
 * Produces format-specific structures that the MLflow UI can parse
 * to render the Chat tab in the trace/span explorer.
 * Includes token usage statistics in the format expected by each provider.
 */
function serializeOutput(output: any, messageFormat: string, usage?: TokenUsage): any {
  if (!output || typeof output !== 'object') {return output;}
  if (output.content === undefined) {
    return output;
  }

  if (messageFormat === 'langchain') {
    // LangChain LLMResult format: {generations: [[{message: {content, type, ...}}]]}
    const result: Record<string, unknown> = {
      generations: [[{
        message: serializeLangchainMessage(output),
      }]],
    };
    if (usage) {
      result.usage_metadata = {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        total_tokens: usage.total_tokens,
      };
    }
    return result;
  }

  if (messageFormat === 'gemini') {
    // Gemini format: {candidates: [{content: {role: "model", parts: [{text: "..."}]}}]}
    const content = output.content;
    const parts = typeof content === 'string'
      ? [{ text: content }]
      : Array.isArray(content) ? content.map((p: any) => ({ text: p.text ?? String(p) })) : [{ text: String(content) }];
    const result: Record<string, unknown> = {
      candidates: [{
        content: { role: 'model', parts },
      }],
    };
    if (usage) {
      result.usageMetadata = {
        promptTokenCount: usage.input_tokens,
        candidatesTokenCount: usage.output_tokens,
        totalTokenCount: usage.total_tokens,
      };
    }
    return result;
  }

  if (messageFormat === 'anthropic') {
    // Anthropic format: {type: "message", role: "assistant", content: [{type: "text", text: "..."}]}
    const contentBlocks: any[] = [];
    const content = output.content;
    if (typeof content === 'string') {
      if (content) {
        contentBlocks.push({ type: 'text', text: content });
      }
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (typeof part === 'string') {
          contentBlocks.push({ type: 'text', text: part });
        } else if (part.type === 'tool_use' || part.type === 'input_json_delta') {
          // Skip tool_use blocks from LangChain's content array - after streaming
          // aggregation, their `input` field is a raw JSON string (not a parsed object),
          // which fails the MLflow UI normalizer's isObject(input) check.
          // Tool calls are instead handled below via output.tool_calls which has
          // properly parsed args objects.
          continue;
        } else if (part.type && MLFLOW_UI_SUPPORTED_CONTENT_TYPES.has(part.type as string)) {
          contentBlocks.push(part);
        } else if (part.type) {
          // Replace unsupported block types (e.g. redacted_thinking) with a text placeholder
          // so the MLflow UI normalizer doesn't reject the entire message
          contentBlocks.push({ type: 'text', text: `[${part.type as string}]` });
        } else {
          contentBlocks.push({ type: 'text', text: part.text ?? String(part) });
        }
      }
    }
    if (output.tool_calls && output.tool_calls.length > 0) {
      for (const tc of output.tool_calls as any[]) {
        contentBlocks.push({
          type: 'tool_use',
          id: tc.id ?? '',
          name: tc.name,
          input: tc.args ?? {},
        });
      }
    }
    const result: Record<string, unknown> = {
      type: 'message',
      role: 'assistant',
      content: contentBlocks,
    };
    if (usage) {
      result.usage = {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
      };
    }
    return result;
  }

  // OpenAI format (default): {choices: [{message: {role, content, tool_calls?}}]}
  const message: Record<string, unknown> = {
    role: 'assistant',
    content: output.content,
  };
  if (output.tool_calls && output.tool_calls.length > 0) {
    message.tool_calls = langchainToolCallsToOpenAI(output.tool_calls);
  }
  const result: Record<string, unknown> = { choices: [{ message }] };
  if (usage) {
    result.usage = {
      prompt_tokens: usage.input_tokens,
      completion_tokens: usage.output_tokens,
      total_tokens: usage.total_tokens,
    };
  }
  return result;
}

/**
 * Extract token usage from a LangChain response.
 * LangChain provides usage_metadata with input_tokens/output_tokens fields.
 */
function extractTokenUsage(response: any): TokenUsage | undefined {
  // LangChain standardized usage_metadata
  const usage = response?.usage_metadata;
  if (!usage) {
    return undefined;
  }

  const inputTokens = usage.input_tokens ?? usage.inputTokens ?? 0;
  const outputTokens = usage.output_tokens ?? usage.outputTokens ?? 0;
  const totalTokens = usage.total_tokens ?? usage.totalTokens ?? (inputTokens + outputTokens);

  return {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: totalTokens,
  };
}
