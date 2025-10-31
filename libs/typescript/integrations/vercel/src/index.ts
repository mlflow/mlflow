import {
  LiveSpan,
  registerOnSpanEndHook,
  registerOnSpanStartHook,
  SpanAttributeKey,
  SpanType
} from 'mlflow-tracing';

const VERCEL_OPERATION_ID_ATTRIBUTE = 'ai.operationId';
const VERCEL_PROMPT_ATTRIBUTE = 'ai.prompt';
const VERCEL_PROMPT_MESSAGES_ATTRIBUTE = 'ai.prompt.messages';
const VERCEL_RESPONSE_TEXT_ATTRIBUTE = 'ai.response.text';
const VERCEL_INPUT_TOKEN_USAGE_ATTRIBUTE = 'ai.usage.promptTokens';
const VERCEL_OUTPUT_TOKEN_USAGE_ATTRIBUTE = 'ai.usage.completionTokens';

const VERCEL_MESSAGE_FORMAT = 'vercel_ai';

export function vercelOnSpanStartHook(span: LiveSpan) {
  if (!isVercelAISpan(span)) {
    return undefined;
  }

  span.setSpanType(SpanType.LLM);
  span.setAttribute(SpanAttributeKey.MESSAGE_FORMAT, VERCEL_MESSAGE_FORMAT);

  const inputs = extractInputs(span);
  if (inputs) {
    span.setInputs(inputs);
  }
}

export function vercelOnSpanEndHook(span: LiveSpan): void {
  if (!isVercelAISpan(span)) {
    return undefined;
  }

  span.allowMutatingEndedSpan = true;

  const outputs = extractOutputs(span);
  if (outputs) {
    span.setOutputs(outputs);
  }

  const tokenUsage = extractTokenUsage(span);
  if (tokenUsage) {
    span.setAttribute(SpanAttributeKey.TOKEN_USAGE, tokenUsage);
  }
}

function isVercelAISpan(span: LiveSpan): boolean {
  return VERCEL_OPERATION_ID_ATTRIBUTE in span.attributes;
}

function extractInputs(span: LiveSpan): Record<string, unknown> | undefined {
  if (!span.attributes) {
    return undefined;
  }

  if (VERCEL_PROMPT_MESSAGES_ATTRIBUTE in span.attributes) {
    return { messages: span.attributes[VERCEL_PROMPT_MESSAGES_ATTRIBUTE] };
  }

  if (VERCEL_PROMPT_ATTRIBUTE in span.attributes) {
    return span.attributes[VERCEL_PROMPT_ATTRIBUTE] as Record<string, unknown>;
  }

  return undefined;
}

function extractOutputs(span: LiveSpan): Record<string, unknown> | undefined {
  if (!span.attributes) {
    return undefined;
  }

  if (VERCEL_RESPONSE_TEXT_ATTRIBUTE in span.attributes) {
    return { text: span.attributes[VERCEL_RESPONSE_TEXT_ATTRIBUTE] as string };
  }

  return undefined;
}

function extractTokenUsage(span: LiveSpan): Record<string, unknown> | undefined {
  if (!span.attributes) {
    return undefined;
  }

  const inputToken = span.attributes[VERCEL_INPUT_TOKEN_USAGE_ATTRIBUTE];
  const outputToken = span.attributes[VERCEL_OUTPUT_TOKEN_USAGE_ATTRIBUTE];
  if (typeof inputToken !== 'number' || typeof outputToken !== 'number') {
    return undefined;
  }
  return {
    input_tokens: inputToken,
    output_tokens: outputToken,
    total_tokens: inputToken + outputToken
  };
}

registerOnSpanStartHook(vercelOnSpanStartHook);
registerOnSpanEndHook(vercelOnSpanEndHook);
