import { describe, expect, it } from '@jest/globals';

import { ModelSpanType } from './ModelTrace.types';
import { inferSpanTypeFromAttributes } from './otlpSpanTypeInference';

describe('inferSpanTypeFromAttributes', () => {
  it('returns CHAT_MODEL when gen_ai.system is present in flat V3 attrs', () => {
    expect(inferSpanTypeFromAttributes({ 'gen_ai.system': 'anthropic' })).toBe(ModelSpanType.CHAT_MODEL);
  });

  it('returns CHAT_MODEL when gen_ai.request.model is present and gen_ai.system is absent', () => {
    expect(inferSpanTypeFromAttributes({ 'gen_ai.request.model': 'claude-opus-4-5' })).toBe(ModelSpanType.CHAT_MODEL);
  });

  it('returns CHAT_MODEL when span.type is llm_request and no gen_ai.* attrs are present', () => {
    expect(inferSpanTypeFromAttributes({ 'span.type': 'llm_request' })).toBe(ModelSpanType.CHAT_MODEL);
  });

  it('returns AGENT when span.type is interaction and no gen_ai.* attrs are present', () => {
    expect(inferSpanTypeFromAttributes({ 'span.type': 'interaction' })).toBe(ModelSpanType.AGENT);
  });

  it('returns undefined when no relevant attributes are present', () => {
    expect(inferSpanTypeFromAttributes({})).toBeUndefined();
  });

  it('returns CHAT_MODEL for V4 array attrs carrying gen_ai.system', () => {
    const v4Attrs = [{ key: 'gen_ai.system', value: { string_value: 'openai' } }];
    expect(inferSpanTypeFromAttributes(v4Attrs)).toBe(ModelSpanType.CHAT_MODEL);
  });

  it('returns CHAT_MODEL (not AGENT) when gen_ai.system and span.type=interaction are both present', () => {
    // gen_ai.* arm is checked before span.type; CHAT_MODEL wins
    expect(inferSpanTypeFromAttributes({ 'gen_ai.system': 'anthropic', 'span.type': 'interaction' })).toBe(
      ModelSpanType.CHAT_MODEL,
    );
  });

  // Encoding-tolerance cases: attribute values may arrive as plain strings or
  // JSON-quoted strings; tryDeserializeAttribute normalises both to the same result.
  it('returns CHAT_MODEL when gen_ai.system is JSON-quoted ("anthropic")', () => {
    expect(inferSpanTypeFromAttributes({ 'gen_ai.system': '"anthropic"' })).toBe(ModelSpanType.CHAT_MODEL);
  });

  it('returns CHAT_MODEL when span.type is JSON-quoted ("llm_request")', () => {
    expect(inferSpanTypeFromAttributes({ 'span.type': '"llm_request"' })).toBe(ModelSpanType.CHAT_MODEL);
  });
});
