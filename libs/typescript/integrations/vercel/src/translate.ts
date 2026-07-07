import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';

// MLflow span type constants (inline to avoid @mlflow/core dependency)
const SPAN_TYPE_LLM = 'LLM';
const SPAN_TYPE_TOOL = 'TOOL';
const SPAN_TYPE_EMBEDDING = 'EMBEDDING';

/**
 * AI SDK operation IDs → MLflow span types.
 *
 * Mirrors the server-side VercelAITranslator mapping in
 * mlflow/tracing/otel/translation/vercel_ai.py.
 */
const OPERATION_ID_TO_SPAN_TYPE: Record<string, string> = {
  'ai.generateText': SPAN_TYPE_LLM,
  'ai.generateText.doGenerate': SPAN_TYPE_LLM,
  'ai.streamText': SPAN_TYPE_LLM,
  'ai.streamText.doStream': SPAN_TYPE_LLM,
  'ai.generateObject': SPAN_TYPE_LLM,
  'ai.generateObject.doGenerate': SPAN_TYPE_LLM,
  'ai.streamObject': SPAN_TYPE_LLM,
  'ai.streamObject.doStream': SPAN_TYPE_LLM,
  'ai.toolCall': SPAN_TYPE_TOOL,
  'ai.embed': SPAN_TYPE_EMBEDDING,
  'ai.embed.doEmbed': SPAN_TYPE_EMBEDDING,
  'ai.embedMany': SPAN_TYPE_EMBEDDING,
  'ai.embedMany.doEmbed': SPAN_TYPE_EMBEDDING,
};

/**
 * Inner "do" operations that carry structured ai.prompt.* / ai.response.* attributes.
 * These represent the actual LLM call within a higher-level operation.
 */
const CHAT_DO_OPERATIONS = new Set([
  'ai.generateText.doGenerate',
  'ai.streamText.doStream',
  'ai.generateObject.doGenerate',
  'ai.streamObject.doStream',
]);

/**
 * Translates Vercel AI SDK span attributes into MLflow's expected format.
 * Mutates span attributes in-place. Spans without `ai.operationId` are left untouched.
 *
 * **No span is ever dropped.** If translation fails for a span, it is left as-is
 * (possibly with partial mlflow.* attrs) and the loop continues.
 *
 * Note: "Call LLM" spans (doGenerate/doStream) already carry gen_ai.* attributes
 * set by the AI SDK (e.g., gen_ai.request.model, gen_ai.usage.input_tokens).
 * The MLflow server's read path handles those. This function adds mlflow.*
 * attributes for span type, structured inputs/outputs, message format, and
 * other features not covered by gen_ai.*.
 */
export function translateSpansForMlflow(spans: ReadableSpan[]): void {
  for (const span of spans) {
    translateSpanForMlflow(span);
  }
}

/**
 * Translates a single Vercel AI SDK span's attributes into MLflow's expected format.
 * Mutates span attributes in-place. Spans without `ai.operationId` are left untouched.
 *
 * If translation fails, the span is left as-is (possibly with partial mlflow.* attrs).
 */
export function translateSpanForMlflow(span: ReadableSpan): void {
  try {
    translateSpan(span);
  } catch (e) {
    console.debug('MLflowSpanProcessor: failed to translate span, passing through unchanged', e);
  }
}

function translateSpan(span: ReadableSpan): void {
  const attrs = span.attributes as Record<string, unknown>;
  const operationId = toStr(attrs['ai.operationId']);
  if (!operationId) {
    return;
  }

  // Span type
  const spanType = OPERATION_ID_TO_SPAN_TYPE[operationId];
  if (spanType && !attrs['mlflow.spanType']) {
    attrs['mlflow.spanType'] = spanType;
  }

  // Span name — use tool name for tool calls instead of generic "ai.toolCall"
  if (operationId === 'ai.toolCall') {
    const toolName = toStr(attrs['ai.toolCall.name']);
    if (toolName) {
      (span as { name: string }).name = toolName;
    }
  }

  // Inputs
  if (!attrs['mlflow.spanInputs']) {
    const inputs = extractInputs(attrs, operationId);
    if (inputs !== undefined) {
      attrs['mlflow.spanInputs'] = safeStringify(inputs);
    }
  }

  // Outputs
  if (!attrs['mlflow.spanOutputs']) {
    const outputs = extractOutputs(attrs, operationId);
    if (outputs !== undefined) {
      attrs['mlflow.spanOutputs'] = safeStringify(outputs);
    }
  }

  // Model (ai.model.id → gen_ai.request.model → gen_ai.response.model)
  if (!attrs['mlflow.llm.model']) {
    const model =
      toStr(attrs['ai.model.id']) ??
      toStr(attrs['gen_ai.request.model']) ??
      toStr(attrs['gen_ai.response.model']);
    if (model) {
      attrs['mlflow.llm.model'] = model;
    }
  }

  // Provider (ai.model.provider → gen_ai.system)
  if (!attrs['mlflow.llm.provider']) {
    const provider = toStr(attrs['ai.model.provider']) ?? toStr(attrs['gen_ai.system']);
    if (provider) {
      attrs['mlflow.llm.provider'] = provider;
    }
  }

  // Message format — only for inner "do" chat spans
  if (!attrs['mlflow.message.format'] && CHAT_DO_OPERATIONS.has(operationId)) {
    attrs['mlflow.message.format'] = 'vercel_ai';
  }

  // Token usage
  if (!attrs['mlflow.chat.tokenUsage']) {
    const tokenUsage = extractTokenUsage(attrs);
    if (tokenUsage !== undefined) {
      attrs['mlflow.chat.tokenUsage'] = tokenUsage;
    }
  }
}

function extractInputs(attrs: Record<string, unknown>, operationId: string): unknown {
  // Chat "do" spans: unpack ai.prompt.* prefix attributes
  if (CHAT_DO_OPERATIONS.has(operationId)) {
    const prefixed = collectPrefix(attrs, 'ai.prompt.');
    if (Object.keys(prefixed).length > 0) {
      return prefixed;
    }
  }

  // Non-chat spans: return first matching raw value (no wrapping)
  const inputKeys = ['ai.prompt', 'ai.toolCall.args', 'ai.value', 'ai.values'];
  for (const attrKey of inputKeys) {
    if (attrs[attrKey] !== undefined) {
      return safeParse(attrs[attrKey]);
    }
  }

  return undefined;
}

function extractOutputs(attrs: Record<string, unknown>, operationId: string): unknown {
  // Chat "do" spans: unpack ai.response.* prefix attributes.
  // This captures keys like ai.response.text and ai.response.object, so
  // the fallback key list below only applies to non-"do" span types.
  if (CHAT_DO_OPERATIONS.has(operationId)) {
    const prefixed = collectPrefix(attrs, 'ai.response.');
    if (Object.keys(prefixed).length > 0) {
      return prefixed;
    }
  }

  // Non-chat spans: return first matching raw value (no wrapping)
  const outputKeys = [
    'ai.response.text',
    'ai.toolCall.result',
    'ai.response.object',
    'ai.embedding',
    'ai.embeddings',
  ];
  for (const attrKey of outputKeys) {
    if (attrs[attrKey] !== undefined) {
      return safeParse(attrs[attrKey]);
    }
  }

  return undefined;
}

/**
 * Extract token usage from ai.usage.* attributes into the mlflow.chat.tokenUsage
 * JSON format: {"input_tokens": N, "output_tokens": N, "total_tokens": N}.
 */
function firstNumber(attrs: Record<string, unknown>, keys: string[]): number | undefined {
  for (const key of keys) {
    const v = toNumber(attrs[key]);
    if (v !== undefined) {
      return v;
    }
  }
  return undefined;
}

function extractTokenUsage(attrs: Record<string, unknown>): string | undefined {
  const inputTokens = firstNumber(attrs, [
    'gen_ai.usage.input_tokens',
    'ai.usage.inputTokens',
    'ai.usage.promptTokens',
  ]);
  const outputTokens = firstNumber(attrs, [
    'gen_ai.usage.output_tokens',
    'ai.usage.outputTokens',
    'ai.usage.completionTokens',
  ]);

  if (inputTokens === undefined && outputTokens === undefined) {
    return undefined;
  }

  const input = inputTokens ?? 0;
  const output = outputTokens ?? 0;
  return safeStringify({
    input_tokens: input,
    output_tokens: output,
    total_tokens: input + output,
  });
}

/**
 * Collect all attributes with a given prefix into an object,
 * stripping the prefix from keys. Values are parsed from JSON if possible.
 */
function collectPrefix(attrs: Record<string, unknown>, prefix: string): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const key of Object.keys(attrs)) {
    if (key.startsWith(prefix)) {
      const shortKey = key.slice(prefix.length);
      result[shortKey] = safeParse(attrs[key]);
    }
  }
  return result;
}

function toStr(value: unknown): string | undefined {
  if (typeof value === 'string' && value.length > 0) {
    return value;
  }
  return undefined;
}

function toNumber(value: unknown): number | undefined {
  if (typeof value === 'number') {
    return value;
  }
  if (typeof value === 'string') {
    const n = Number(value);
    return Number.isFinite(n) ? n : undefined;
  }
  return undefined;
}

/**
 * Safely parse a value that may be a JSON string (possibly double-encoded from OTLP).
 * Attempts up to 2 levels of JSON decoding.
 */
function safeParse(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => safeParse(item));
  }
  if (typeof value !== 'string') {
    return value;
  }
  try {
    const first = JSON.parse(value) as unknown;
    // Try one more level of decoding for double-encoded values
    if (typeof first === 'string') {
      try {
        return JSON.parse(first) as unknown;
      } catch {
        return first;
      }
    }
    return first;
  } catch {
    return value;
  }
}

function safeStringify(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}
