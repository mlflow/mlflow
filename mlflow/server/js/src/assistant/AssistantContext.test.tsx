import { describe, test, expect } from '@jest/globals';
import { upsertToolCalls, applyToolResult, reviveMessages, trimForStorage } from './AssistantContext';
import type { AssistantPart, ChatMessage } from './types';

describe('upsertToolCalls', () => {
  test('appends a new tool call with running status', () => {
    const result = upsertToolCalls([], [{ id: 't1', name: 'Bash', input: { command: 'ls' } }]);
    expect(result).toEqual([
      { type: 'toolCall', toolUseId: 't1', name: 'Bash', input: { command: 'ls' }, status: 'running' },
    ]);
  });

  test('keeps a tool call after any text part', () => {
    const parts: AssistantPart[] = [{ type: 'text', text: 'working' }];
    const result = upsertToolCalls(parts, [{ id: 't1', name: 'Bash', input: {} }]);
    expect(result).toHaveLength(2);
    expect(result[0]).toEqual({ type: 'text', text: 'working' });
    expect(result[1]).toMatchObject({ type: 'toolCall', toolUseId: 't1', status: 'running' });
  });

  test('re-upserting an existing call does not clobber a resolved status/result', () => {
    const parts: AssistantPart[] = [
      { type: 'toolCall', toolUseId: 't1', name: 'Bash', input: { command: 'ls' }, status: 'done', result: 'out' },
    ];
    const result = upsertToolCalls(parts, [{ id: 't1', name: 'Bash', input: { command: 'ls -a' } }]);
    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({ status: 'done', result: 'out', input: { command: 'ls -a' } });
  });
});

describe('applyToolResult', () => {
  const parts: AssistantPart[] = [
    { type: 'text', text: 'let me check' },
    { type: 'toolCall', toolUseId: 't1', name: 'Bash', input: { command: 'ls' }, status: 'running' },
  ];

  test('resolves the matching tool call to done with its result', () => {
    const result = applyToolResult(parts, { toolUseId: 't1', content: 'output', isError: false });
    expect(result[1]).toMatchObject({ toolUseId: 't1', status: 'done', result: 'output' });
    expect(result[0]).toEqual({ type: 'text', text: 'let me check' });
  });

  test('marks the tool call as error when isError is true', () => {
    const result = applyToolResult(parts, { toolUseId: 't1', content: 'boom', isError: true });
    expect(result[1]).toMatchObject({ status: 'error', result: 'boom' });
  });

  test('leaves parts untouched when no toolUseId matches', () => {
    const result = applyToolResult(parts, { toolUseId: 'other', content: 'x', isError: false });
    expect(result).toEqual(parts);
  });
});

describe('reviveMessages', () => {
  test('revives string timestamps (from JSON) into Date objects', () => {
    const revived = reviveMessages([
      { id: 'm1', role: 'assistant', content: 'hi', timestamp: '2026-06-17T12:00:00.000Z' } as unknown as ChatMessage,
    ]);
    expect(revived[0].timestamp).toBeInstanceOf(Date);
    expect(revived[0].timestamp.getTime()).toBe(new Date('2026-06-17T12:00:00.000Z').getTime());
  });

  test('returns an empty array unchanged', () => {
    expect(reviveMessages([])).toEqual([]);
  });
});

describe('trimForStorage', () => {
  const at = new Date('2026-01-01T00:00:00.000Z');
  const msg = (id: string, content: string): ChatMessage => ({
    id,
    role: 'assistant',
    content,
    timestamp: at,
  });

  test('leaves a small conversation unchanged', () => {
    const messages = [msg('m1', 'hello'), msg('m2', 'world')];
    expect(trimForStorage(messages)).toHaveLength(2);
  });

  test('drops oldest messages but always keeps the most recent', () => {
    const messages = [msg('m1', 'a'), msg('m2', 'b'), msg('m3', 'c')];
    const trimmed = trimForStorage(messages, 10);
    expect(trimmed).toHaveLength(1);
    expect(trimmed[0].id).toBe('m3');
  });

  test('keeps as many recent messages as fit the byte budget', () => {
    const messages = [msg('m1', 'a'), msg('m2', 'b'), msg('m3', 'c')];
    const budgetForLastTwo = JSON.stringify(messages.slice(1)).length;
    const trimmed = trimForStorage(messages, budgetForLastTwo);
    expect(trimmed.map((m) => m.id)).toEqual(['m2', 'm3']);
  });
});
