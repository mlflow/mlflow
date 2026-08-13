import { describe, it, expect } from '@jest/globals';
import type { ChatMessage } from './types';
import {
  extractTemplateVariables,
  formatJson,
  getEmptyVariables,
  getToolParametersError,
  prettyPrintJson,
  substituteVariables,
} from './utils';

describe('extractTemplateVariables', () => {
  it('returns an empty list when there are no placeholders', () => {
    const messages: ChatMessage[] = [
      { role: 'system', content: 'You are concise.' },
      { role: 'user', content: 'Hello there' },
    ];
    expect(extractTemplateVariables(messages)).toEqual([]);
  });

  it('extracts a single variable', () => {
    const messages: ChatMessage[] = [{ role: 'user', content: 'Summarize: {{ text }}' }];
    expect(extractTemplateVariables(messages)).toEqual(['text']);
  });

  it('preserves order of first appearance and de-dupes', () => {
    const messages: ChatMessage[] = [
      { role: 'system', content: 'Style: {{ tone }}' },
      { role: 'user', content: 'Translate {{ text }} into {{ language }}; keep tone {{ tone }}.' },
    ];
    expect(extractTemplateVariables(messages)).toEqual(['tone', 'text', 'language']);
  });

  it('ignores malformed placeholders (no inner identifier)', () => {
    const messages: ChatMessage[] = [{ role: 'user', content: 'Try {{ }} or {{1bad}}' }];
    expect(extractTemplateVariables(messages)).toEqual([]);
  });
});

describe('substituteVariables', () => {
  it('replaces every occurrence of a placeholder with the provided value', () => {
    const messages: ChatMessage[] = [{ role: 'user', content: '{{ text }} appears twice: {{ text }}' }];
    expect(substituteVariables(messages, { text: 'hi' })).toEqual([{ role: 'user', content: 'hi appears twice: hi' }]);
  });

  it('uses an empty string for variables without a value', () => {
    const messages: ChatMessage[] = [{ role: 'user', content: '[{{ a }}][{{ b }}]' }];
    expect(substituteVariables(messages, { a: 'X' })).toEqual([{ role: 'user', content: '[X][]' }]);
  });

  it('preserves roles and order, and leaves messages without placeholders unchanged', () => {
    const messages: ChatMessage[] = [
      { role: 'system', content: 'You are concise.' },
      { role: 'user', content: 'Summarize: {{ text }}' },
    ];
    expect(substituteVariables(messages, { text: 'Hello world' })).toEqual([
      { role: 'system', content: 'You are concise.' },
      { role: 'user', content: 'Summarize: Hello world' },
    ]);
  });

  it('leaves malformed placeholders literal', () => {
    const messages: ChatMessage[] = [{ role: 'user', content: 'Keep {{ }} as-is' }];
    expect(substituteVariables(messages, {})).toEqual([{ role: 'user', content: 'Keep {{ }} as-is' }]);
  });
});

describe('getEmptyVariables', () => {
  it('returns an empty list when no variables are declared', () => {
    const messages: ChatMessage[] = [{ role: 'user', content: 'no placeholders' }];
    expect(getEmptyVariables(messages, {})).toEqual([]);
  });

  it('returns every declared variable when no values are provided', () => {
    const messages: ChatMessage[] = [{ role: 'user', content: '{{ topic }} in {{ tone }}' }];
    expect(getEmptyVariables(messages, {})).toEqual(['topic', 'tone']);
  });

  it('returns only the variables whose value is missing or trims to empty', () => {
    const messages: ChatMessage[] = [{ role: 'user', content: '{{ a }} {{ b }} {{ c }}' }];
    expect(getEmptyVariables(messages, { a: 'set', b: '   ', c: '' })).toEqual(['b', 'c']);
  });

  it('preserves first-appearance order from extractTemplateVariables', () => {
    const messages: ChatMessage[] = [
      { role: 'system', content: 'Style: {{ tone }}' },
      { role: 'user', content: 'Translate {{ text }} into {{ language }}; keep tone {{ tone }}.' },
    ];
    expect(getEmptyVariables(messages, { language: 'fr' })).toEqual(['tone', 'text']);
  });
});

describe('formatJson', () => {
  it('pretty-prints valid JSON with 2-space indentation', () => {
    expect(formatJson('{"a":1,"b":[2,3]}')).toBe('{\n  "a": 1,\n  "b": [\n    2,\n    3\n  ]\n}');
  });

  it('returns null for invalid JSON', () => {
    expect(formatJson('{not json')).toBeNull();
    expect(formatJson('')).toBeNull();
  });
});

describe('getToolParametersError', () => {
  it('flags empty or whitespace-only text', () => {
    expect(getToolParametersError('')).toEqual({ code: 'empty' });
    expect(getToolParametersError('   ')).toEqual({ code: 'empty' });
    expect(getToolParametersError('\n\t')).toEqual({ code: 'empty' });
  });

  it('returns a parse error with the parser detail for unparseable text', () => {
    const result = getToolParametersError('{not-json');
    expect(result?.code).toBe('parseError');
    expect(result).toMatchObject({ code: 'parseError', detail: expect.any(String) });
  });

  it('flags JSON that is not an object', () => {
    expect(getToolParametersError('[]')).toEqual({ code: 'notObject' });
    expect(getToolParametersError('"foo"')).toEqual({ code: 'notObject' });
    expect(getToolParametersError('42')).toEqual({ code: 'notObject' });
    expect(getToolParametersError('null')).toEqual({ code: 'notObject' });
  });

  it('flags an object schema without a properties map', () => {
    expect(getToolParametersError('{}')).toEqual({ code: 'missingProperties' });
    expect(getToolParametersError('{"type":"object"}')).toEqual({ code: 'missingProperties' });
    expect(getToolParametersError('{"properties":[]}')).toEqual({ code: 'missingProperties' });
  });

  it('returns null for a valid object schema with properties', () => {
    expect(getToolParametersError('{"type":"object","properties":{}}')).toBeNull();
    expect(getToolParametersError('{"properties":{"a":{"type":"string"}}}')).toBeNull();
  });
});

describe('prettyPrintJson', () => {
  it('pretty-prints valid JSON with 2-space indentation', () => {
    expect(prettyPrintJson('{"city":"San Francisco"}')).toBe('{\n  "city": "San Francisco"\n}');
  });

  it('re-stringifies regardless of input whitespace so output is canonical', () => {
    expect(prettyPrintJson('{ "a" : 1 ,"b":2 }')).toBe('{\n  "a": 1,\n  "b": 2\n}');
  });

  it('falls back to the raw string for invalid/partial JSON', () => {
    expect(prettyPrintJson('{"city":"San Francisco"')).toBe('{"city":"San Francisco"');
    expect(prettyPrintJson('not json at all')).toBe('not json at all');
  });
});
