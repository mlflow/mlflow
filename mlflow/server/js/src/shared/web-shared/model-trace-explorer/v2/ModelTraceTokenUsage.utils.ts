import type { TokenUsage } from '../ModelTraceExplorerTokenUsageHoverCard';

export type SpanTokenUsage = Partial<TokenUsage> & Pick<TokenUsage, 'total_tokens'>;

export interface SpanTokenUsageSource {
  attributes?: unknown;
  tokenUsage?: SpanTokenUsage;
}

const MLFLOW_TOKEN_USAGE_ATTRIBUTE = 'mlflow.chat.tokenUsage';
const OTEL_INPUT_TOKENS_ATTRIBUTE = 'gen_ai.usage.input_tokens';
const OTEL_OUTPUT_TOKENS_ATTRIBUTE = 'gen_ai.usage.output_tokens';
const OTEL_LEGACY_INPUT_TOKENS_ATTRIBUTE = 'gen_ai.usage.prompt_tokens';
const OTEL_LEGACY_OUTPUT_TOKENS_ATTRIBUTE = 'gen_ai.usage.completion_tokens';
const OTEL_CACHE_READ_INPUT_TOKENS_ATTRIBUTE = 'gen_ai.usage.cache_read_input_tokens';
const OTEL_CACHE_CREATION_INPUT_TOKENS_ATTRIBUTE = 'gen_ai.usage.cache_creation_input_tokens';

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value);

const getAttribute = (attributes: unknown, key: string): unknown => {
  if (isRecord(attributes)) {
    return attributes[key];
  }

  if (!Array.isArray(attributes)) {
    return undefined;
  }

  const attribute = attributes.find((candidate) => isRecord(candidate) && candidate['key'] === key);
  if (!isRecord(attribute) || !isRecord(attribute['value'])) {
    return undefined;
  }

  const value = attribute['value'];
  return value['string_value'] ?? value['int_value'] ?? value['bool_value'];
};

const getFiniteNumber = (value: unknown): number | undefined => {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : undefined;
  }

  if (typeof value !== 'string' || value.trim() === '') {
    return undefined;
  }

  const parsedValue = Number(value);
  return Number.isFinite(parsedValue) ? parsedValue : undefined;
};

const getNumberField = (value: Record<string, unknown>, field: string): number | undefined =>
  getFiniteNumber(value[field]);

const normalizeMlflowTokenUsage = (value: unknown): SpanTokenUsage | undefined => {
  let parsedValue = value;
  if (typeof parsedValue === 'string') {
    try {
      parsedValue = JSON.parse(parsedValue);
    } catch {
      return undefined;
    }
  }

  if (!isRecord(parsedValue)) {
    return undefined;
  }

  let inputTokens = getNumberField(parsedValue, 'input_tokens');
  let outputTokens = getNumberField(parsedValue, 'output_tokens');
  const totalTokens =
    getNumberField(parsedValue, 'total_tokens') ??
    (inputTokens !== undefined && outputTokens !== undefined ? inputTokens + outputTokens : undefined);

  if (totalTokens === undefined) {
    return undefined;
  }

  if (inputTokens === undefined && outputTokens !== undefined) {
    inputTokens = Math.max(totalTokens - outputTokens, 0);
  }
  if (outputTokens === undefined && inputTokens !== undefined) {
    outputTokens = Math.max(totalTokens - inputTokens, 0);
  }

  const cacheReadInputTokens = getNumberField(parsedValue, 'cache_read_input_tokens');
  const cacheCreationInputTokens = getNumberField(parsedValue, 'cache_creation_input_tokens');

  return {
    total_tokens: totalTokens,
    ...(inputTokens !== undefined ? { input_tokens: inputTokens } : {}),
    ...(outputTokens !== undefined ? { output_tokens: outputTokens } : {}),
    ...(cacheReadInputTokens !== undefined ? { cache_read_input_tokens: cacheReadInputTokens } : {}),
    ...(cacheCreationInputTokens !== undefined ? { cache_creation_input_tokens: cacheCreationInputTokens } : {}),
  };
};

const getOtelTokenUsage = (attributes: unknown): SpanTokenUsage | undefined => {
  const inputTokens =
    getFiniteNumber(getAttribute(attributes, OTEL_INPUT_TOKENS_ATTRIBUTE)) ??
    getFiniteNumber(getAttribute(attributes, OTEL_LEGACY_INPUT_TOKENS_ATTRIBUTE));
  const outputTokens =
    getFiniteNumber(getAttribute(attributes, OTEL_OUTPUT_TOKENS_ATTRIBUTE)) ??
    getFiniteNumber(getAttribute(attributes, OTEL_LEGACY_OUTPUT_TOKENS_ATTRIBUTE));

  if (inputTokens === undefined || outputTokens === undefined) {
    return undefined;
  }

  const cacheReadInputTokens = getFiniteNumber(getAttribute(attributes, OTEL_CACHE_READ_INPUT_TOKENS_ATTRIBUTE));
  const cacheCreationInputTokens = getFiniteNumber(
    getAttribute(attributes, OTEL_CACHE_CREATION_INPUT_TOKENS_ATTRIBUTE),
  );

  return {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: inputTokens + outputTokens,
    ...(cacheReadInputTokens !== undefined ? { cache_read_input_tokens: cacheReadInputTokens } : {}),
    ...(cacheCreationInputTokens !== undefined ? { cache_creation_input_tokens: cacheCreationInputTokens } : {}),
  };
};

export const getSpanTokenUsage = (span: SpanTokenUsageSource): SpanTokenUsage | undefined =>
  span.tokenUsage ??
  normalizeMlflowTokenUsage(getAttribute(span.attributes, MLFLOW_TOKEN_USAGE_ATTRIBUTE)) ??
  getOtelTokenUsage(span.attributes);
