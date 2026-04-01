import { describe, it, expect } from '@jest/globals';

import { ModelSpanType, type ModelTraceSpanNode } from '../ModelTrace.types';
import { createListFromObject } from '../ModelTraceExplorer.utils';
import { spanMatchesFilter, applyJsonPath, applyJsonPathToObject } from './useTraceViewFiltering';

const makeSpanNode = (overrides: Partial<ModelTraceSpanNode> = {}): ModelTraceSpanNode => ({
  key: 'span-1',
  title: 'plan_action',
  type: ModelSpanType.FUNCTION,
  start: 0,
  end: 100,
  inputs: { message: 'hello' },
  outputs: { reasoning: 'think step by step', query: 'search docs', model: 'gpt-4o' },
  attributes: { 'mlflow.spanType': 'FUNCTION', customAttr: 'customValue' },
  assessments: [],
  traceId: 'tr-1',
  ...overrides,
});

describe('spanMatchesFilter', () => {
  it('returns true when filter is null', () => {
    expect(spanMatchesFilter(makeSpanNode(), null)).toBe(true);
  });

  it('returns true when filter is undefined', () => {
    expect(spanMatchesFilter(makeSpanNode(), undefined)).toBe(true);
  });

  it('matches by span_name', () => {
    const span = makeSpanNode({ title: 'plan_action' });
    expect(spanMatchesFilter(span, { span_name: 'plan_action' })).toBe(true);
    expect(spanMatchesFilter(span, { span_name: 'other_span' })).toBe(false);
  });

  it('matches by span_type (case insensitive)', () => {
    const span = makeSpanNode({ type: ModelSpanType.TOOL });
    expect(spanMatchesFilter(span, { span_type: 'TOOL' })).toBe(true);
    expect(spanMatchesFilter(span, { span_type: 'tool' })).toBe(true);
    expect(spanMatchesFilter(span, { span_type: 'LLM' })).toBe(false);
  });

  it('falls back to mlflow.spanType attribute when type is null', () => {
    const span = makeSpanNode({
      type: undefined,
      attributes: { 'mlflow.spanType': 'RETRIEVER' },
    });
    expect(spanMatchesFilter(span, { span_type: 'RETRIEVER' })).toBe(true);
    expect(spanMatchesFilter(span, { span_type: 'TOOL' })).toBe(false);
  });

  it('matches by attribute_key existence', () => {
    const span = makeSpanNode({ attributes: { customAttr: 'val' } });
    expect(spanMatchesFilter(span, { attribute_key: 'customAttr' })).toBe(true);
    expect(spanMatchesFilter(span, { attribute_key: 'missing' })).toBe(false);
  });

  it('matches by attribute_key + attribute_value', () => {
    const span = makeSpanNode({ attributes: { model: 'gpt-4o' } });
    expect(spanMatchesFilter(span, { attribute_key: 'model', attribute_value: 'gpt-4o' })).toBe(true);
    expect(spanMatchesFilter(span, { attribute_key: 'model', attribute_value: 'claude' })).toBe(false);
  });

  it('requires all filter fields to match (AND logic)', () => {
    const span = makeSpanNode({ title: 'plan_action', type: ModelSpanType.FUNCTION });
    expect(spanMatchesFilter(span, { span_name: 'plan_action', span_type: 'FUNCTION' })).toBe(true);
    expect(spanMatchesFilter(span, { span_name: 'plan_action', span_type: 'TOOL' })).toBe(false);
  });

  it('returns false when attributes is an array', () => {
    const span = makeSpanNode({ attributes: ['not', 'a', 'map'] as any });
    expect(spanMatchesFilter(span, { attribute_key: 'anything' })).toBe(false);
  });
});

describe('applyJsonPath', () => {
  it('returns original data when jsonPath is null', () => {
    expect(applyJsonPath('{"x": 1}', null)).toBe('{"x": 1}');
  });

  it('returns original data when jsonPath is undefined', () => {
    expect(applyJsonPath('{"x": 1}', undefined)).toBe('{"x": 1}');
  });

  it('extracts a simple field from a JSON string', () => {
    const data = JSON.stringify({ message: 'hello', extra: 'stuff' });
    const result = applyJsonPath(data, '$.message');
    expect(JSON.parse(result)).toBe('hello');
  });

  it('extracts nested fields', () => {
    const data = JSON.stringify({ response: { text: 'answer' } });
    const result = applyJsonPath(data, '$.response.text');
    expect(JSON.parse(result)).toBe('answer');
  });

  it('returns original data when path matches nothing', () => {
    const data = JSON.stringify({ foo: 'bar' });
    expect(applyJsonPath(data, '$.missing')).toBe(data);
  });

  it('returns original data for invalid JSON input', () => {
    expect(applyJsonPath('not json', '$.x')).toBe('not json');
  });

  it('returns array for multiple matches', () => {
    const data = JSON.stringify({ items: [{ v: 1 }, { v: 2 }] });
    const result = applyJsonPath(data, '$.items[*].v');
    expect(JSON.parse(result)).toEqual([1, 2]);
  });
});

describe('applyJsonPathToObject', () => {
  it('returns original data when jsonPath is null', () => {
    const obj = { reasoning: 'think', query: 'search' };
    expect(applyJsonPathToObject(obj, null)).toBe(obj);
  });

  it('returns original data when jsonPath is undefined', () => {
    const obj = { reasoning: 'think' };
    expect(applyJsonPathToObject(obj, undefined)).toBe(obj);
  });

  it('returns original data when data is null', () => {
    expect(applyJsonPathToObject(null, '$.x')).toBeNull();
  });

  it('returns original data when data is undefined', () => {
    expect(applyJsonPathToObject(undefined, '$.x')).toBeUndefined();
  });

  it('extracts a top-level key from an object', () => {
    const obj = {
      reasoning: 'User wants a refund',
      query: 'refund policy',
      model: 'gpt-4o',
      usage: { prompt_tokens: 100 },
    };
    expect(applyJsonPathToObject(obj, '$.reasoning')).toBe('User wants a refund');
  });

  it('extracts nested values', () => {
    const obj = { response: { text: 'answer', metadata: {} } };
    expect(applyJsonPathToObject(obj, '$.response.text')).toBe('answer');
  });

  it('returns original data when path matches nothing', () => {
    const obj = { foo: 'bar' };
    expect(applyJsonPathToObject(obj, '$.missing')).toBe(obj);
  });

  it('returns array for multiple matches', () => {
    const obj = { messages: [{ content: 'a' }, { content: 'b' }] };
    expect(applyJsonPathToObject(obj, '$.messages[*].content')).toEqual(['a', 'b']);
  });

  it('returns the single value unwrapped for single match', () => {
    const obj = { items: [{ v: 42 }] };
    expect(applyJsonPathToObject(obj, '$.items[0].v')).toBe(42);
  });

  it('works with the exact trace view scenario that was buggy', () => {
    // This is the real-world case: a plan_action span's outputs object
    // with output_path "$.reasoning" should extract just the reasoning string.
    const outputs = {
      messages: [
        { role: 'system', content: 'You are a planning agent.' },
        { role: 'user', content: 'Can I get a refund on my international order?' },
      ],
      query: 'refund policy international orders',
      reasoning: 'User is asking about refunds for international orders. Need to search the refund policy docs.',
      model: 'gpt-4o-mini',
      usage: { prompt_tokens: 234, completion_tokens: 89 },
    };

    const result = applyJsonPathToObject(outputs, '$.reasoning');
    expect(result).toBe(
      'User is asking about refunds for international orders. Need to search the refund policy docs.',
    );
  });

  it('returns original data on invalid JSONPath', () => {
    const obj = { x: 1 };
    // Invalid path should not throw, just return original
    expect(applyJsonPathToObject(obj, '$[invalid[')).toBe(obj);
  });
});

describe('applyJsonPathToObject + createListFromObject integration', () => {
  it('filters outputs to a single field before list conversion', () => {
    // This tests the exact pipeline used in the summary and detail views:
    // raw span outputs → applyJsonPathToObject → createListFromObject → rendered list
    const outputs = {
      messages: [{ role: 'system', content: 'You are a planning agent.' }],
      query: 'refund policy international orders',
      reasoning: 'User is asking about refunds.',
      model: 'gpt-4o-mini',
      usage: { prompt_tokens: 234, completion_tokens: 89 },
    };

    // Without filtering: all 5 keys show
    const unfilteredList = createListFromObject(outputs);
    expect(unfilteredList).toHaveLength(5);
    expect(unfilteredList.map((item) => item.key)).toEqual(['messages', 'query', 'reasoning', 'model', 'usage']);

    // With $.reasoning filter: only the reasoning value shows
    const filtered = applyJsonPathToObject(outputs, '$.reasoning');
    const filteredList = createListFromObject(filtered as any);
    expect(filteredList).toHaveLength(1);
    expect(filteredList[0].value).toBe('"User is asking about refunds."');
  });

  it('filters inputs with a nested JSONPath', () => {
    const inputs = {
      messages: [
        { role: 'user', content: 'What is the refund policy?' },
        { role: 'assistant', content: 'Let me check...' },
      ],
      config: { temperature: 0.7 },
    };

    const filtered = applyJsonPathToObject(inputs, '$.messages[*].content');
    const list = createListFromObject(filtered as any);
    // Should produce a list from the extracted array
    expect(list).toHaveLength(1);
    const parsed = JSON.parse(list[0].value);
    expect(parsed).toEqual(['What is the refund policy?', 'Let me check...']);
  });

  it('preserves all data when JSONPath is null (no view active)', () => {
    const outputs = { reasoning: 'think', query: 'search' };
    const result = applyJsonPathToObject(outputs, null);
    expect(result).toBe(outputs);
    const list = createListFromObject(result as any);
    expect(list).toHaveLength(2);
  });
});
