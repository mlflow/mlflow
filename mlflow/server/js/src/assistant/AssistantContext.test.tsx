import { describe, test, expect } from '@jest/globals';
import { upsertToolCalls, applyToolResult } from './AssistantContext';
import type { AssistantPart } from './types';

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
