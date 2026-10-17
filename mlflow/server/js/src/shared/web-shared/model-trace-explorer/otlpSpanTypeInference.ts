import { ModelSpanType } from './ModelTrace.types';
import type { ModelTraceSpan } from './ModelTrace.types';
import { getSpanAttribute } from './spanAttributeReader';
import { tryDeserializeAttribute } from './spanAttributeDeserializer';

/**
 * Infers a display-only span type from raw OTel attributes when mlflow.spanType is absent.
 * Producers that set mlflow.spanType (e.g. the MLflow SDK) are unaffected — the caller
 * must treat the result of this function only as a fallback.
 *
 * Each attribute value is run through tryDeserializeAttribute so that both plain
 * strings ("llm_request") and JSON-quoted strings ('"llm_request"') are handled.
 *
 * Precedence:
 *   1. Any OTel GenAI semantic-convention attribute (gen_ai.system, gen_ai.request.model) → CHAT_MODEL
 *   2. Claude Code span.type==='llm_request' → CHAT_MODEL
 *   3. Claude Code span.type==='interaction' → AGENT
 *   4. Otherwise → undefined (caller resolves to UNKNOWN)
 */
export const inferSpanTypeFromAttributes = (attributes: ModelTraceSpan['attributes']): ModelSpanType | undefined => {
  const genAiSystem = getSpanAttribute(attributes, 'gen_ai.system');
  const genAiRequestModel = getSpanAttribute(attributes, 'gen_ai.request.model');
  if (
    (typeof genAiSystem === 'string' && tryDeserializeAttribute(genAiSystem) !== undefined) ||
    (typeof genAiRequestModel === 'string' && tryDeserializeAttribute(genAiRequestModel) !== undefined)
  ) {
    return ModelSpanType.CHAT_MODEL;
  }

  const rawCcSpanType = getSpanAttribute(attributes, 'span.type');
  const ccSpanType = typeof rawCcSpanType === 'string' ? tryDeserializeAttribute(rawCcSpanType) : undefined;
  if (ccSpanType === 'llm_request') {
    return ModelSpanType.CHAT_MODEL;
  }
  if (ccSpanType === 'interaction') {
    return ModelSpanType.AGENT;
  }

  return undefined;
};
