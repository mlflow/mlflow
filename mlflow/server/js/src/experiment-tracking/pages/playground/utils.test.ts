import { describe, it, expect } from '@jest/globals';
import type { ChatMessage } from './types';
import { extractTemplateVariables, getEmptyVariables, isToolsValueEmpty, substituteVariables } from './utils';

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

describe('isToolsValueEmpty', () => {
  it('treats empty and whitespace-only text as empty', () => {
    expect(isToolsValueEmpty('')).toBe(true);
    expect(isToolsValueEmpty('   ')).toBe(true);
    expect(isToolsValueEmpty('\n\t')).toBe(true);
  });

  it('treats an empty JSON array as empty', () => {
    expect(isToolsValueEmpty('[]')).toBe(true);
    expect(isToolsValueEmpty('[ ]')).toBe(true);
    expect(isToolsValueEmpty('[\n]')).toBe(true);
  });

  it('treats a non-empty array as not empty', () => {
    expect(isToolsValueEmpty('[{}]')).toBe(false);
    expect(isToolsValueEmpty('[{"type":"function"}]')).toBe(false);
  });

  it('returns false for non-array JSON so the parse-error path can claim it', () => {
    expect(isToolsValueEmpty('{}')).toBe(false);
    expect(isToolsValueEmpty('"foo"')).toBe(false);
    expect(isToolsValueEmpty('42')).toBe(false);
  });

  it('returns false for unparseable text so the parse-error path can claim it', () => {
    expect(isToolsValueEmpty('[not-json')).toBe(false);
    expect(isToolsValueEmpty('not-json')).toBe(false);
  });
});
